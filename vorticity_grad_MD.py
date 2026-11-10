openmm_env_path='/home/adrijad2/om77'

import os
import sys
#sys.path.insert(0,openmm_env_path+'lib/python3.10/site-packages')
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from simtk.openmm import XmlSerializer
from parmed.amber import LoadParm            #Loads the parmed library we just installed.
from parmed.openmm import RestartReporter
from parmed import unit as u
from mdtraj.reporters import DCDReporter
from openmmtools import states
import re
import numpy as np
from openmm.unit import kilojoule_per_mole, angstrom
from openmm import CustomExternalForce, CustomCentroidBondForce, CustomBondForce, CustomAngleForce, CustomTorsionForce


def flow(parm_name, system_name): 

    #COMPUTING-PLATFORM--------------------------------------------------------------
    
    ##how this simulation will be run
    
    ##Single GPU
    platform = Platform.getPlatformByName('CUDA')
    platformProperties = {'CudaPrecision': 'single'}
    
    ##If multiple GPUs
    #platformProperties = {'CudaPrecision': 'mixed', 'CUDADeviceIndex': '0,1,2,3'}
    
    ##CPUs
    #platform = Platform.getPlatformByName('CPU')
    #platformProperties = {'CpuThreads': 'default'}
    
    #SIMULATION-INPUTS---------------------------------------------------------------
    
    ##variables related to barostat, thermostat, and force calculations
    
    constraints = HBonds                                   #Puts constraints on
    rigidWater = True                                      #Makes water molecules rigid
    constraintTolerance = 0.000001                         #Tolerance for constraints
    temperature = 300*kelvin                               #Temperature in Kelvin
    friction = 2.8284/picosecond                           #Collision frequency for Langevin thermostat
    
    ##using Chamber input files
    print('Reading parameters')
    parm = LoadParm(parm_name, system_name+'_EQ_fbb.rst7')
    print('Creating system')


    topology = parm.topology                               #Could be psf.topology if done with CHARMM
    positions = parm.positions
    
    print('Building system...')
    system = parm.createSystem(nonbondedMethod=NoCutoff,constraints=constraints, 
                                implicitSolvent=OBC2, soluteDielectric=1.0,
                                solventDielectric=5.621)

    ##Generate initil xml file
    f = open(system_name+'_EQ.xml', 'w+')
    f.write(XmlSerializer.serialize(system))
    f.close()
    
    ##SIMULATION-PARAMETERS-----------------------------------------------------------

    ##assumes you have already done hydrogen mass repartioning somewhere before
    ##else, when you create you 'system', include parm/psf.createSystem(hydrogenMass=4*amu)

    #Determine simulation length
    dt = 0.002*picoseconds        #Timestep. 4 fs assuming HMR and Verlet
    #integratorPressure for simulations
    n_steps = 25000000            #-->100ns reducing to 50 ns                #Simulation length. 250000 = 1 ns at 4fs dt


    #INTEGRATOR----------------------------------------------------------------------
    
    print('Creating integrator...')
    integrator = LangevinIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)

    
    ######## FORCES ############################################################
    coords = np.asarray(parm.coordinates) * 0.1  # to convert ang to nm
    polymer_bb_atoms = []
    indices_edge = [76, 6818, 10376, 10522] # atom indices of some of the edge atoms 
    POL_RES = ['MOH', 'MOM', 'MOT']
    MON_BB_ATOM = ['C05','C07','C09','C12','C18', 'C21']
    
    for res in parm.residues:
        if res.name in POL_RES:
            for atom in res.atoms:
                if atom.name in MON_BB_ATOM:
                    polymer_bb_atoms.append(atom.idx)

            
    polymer_coords = coords[polymer_bb_atoms]
    xc, yc, zc = polymer_coords.mean(axis=0)        # switching to the centre of the polymer instead of the box 
    polymer_x = np.array([coords[i][0] for i in polymer_bb_atoms])
    x_min = polymer_x.min()
    x_max = polymer_x.max()
    x_length = x_max - x_min
    
    ## add a rotating force but make it monotonically increasing
    ## this is the rotating force 
    energy = "0.5*k*((x-x0)^2 + (y-yref)^2 + (z-zref)^2); yref = yc + r0*cos(theta0 + omega_scale*dphi); zref = zc + r0*sin(theta0 + omega_scale*dphi)"      #yc and zc are the y,z of the box center
    rotating_force = CustomExternalForce(energy)
    rotating_force.addGlobalParameter("yc", yc*nanometer) 
    rotating_force.addGlobalParameter("zc", zc*nanometer)
    rotating_force.addGlobalParameter("k", 10.0 *kilojoule_per_mole/nanometer**2)   # previous value 25.0
    rotating_force.addGlobalParameter("dphi", 0.0)
    rotating_force.addPerParticleParameter("r0")      ## initial orbital radius from the center of the box
    rotating_force.addPerParticleParameter("theta0")  ## initial angular position
    rotating_force.addPerParticleParameter("x0")      ## initial x coordinate, so the particle rotates about the x-axis 
    rotating_force.addPerParticleParameter("omega_scale") #


    # add harmonic restrain to eh the first and the last carbon of the polymer chain 
    force_edge = CustomExternalForce("k_e*periodicdistance(x, y, z, x0, y0, z0)^2")
    force_edge.addGlobalParameter("k_e", 0.05*kilocalories_per_mole/angstroms**2)
    force_edge.addPerParticleParameter("x0")
    force_edge.addPerParticleParameter("y0")
    force_edge.addPerParticleParameter("z0")

    
    ### add the forces to the desired atom groups #############################
    debug_r0 = None
    debug_theta0 = None
    debug_x0 = None
    
               
    for res in parm.residues:
        if res.name in POL_RES:       
            for atom in res.atoms:
                if atom.name in MON_BB_ATOM:
                    atom_id = atom.idx
                    x_0, y_0, z_0 = coords[atom_id]
                    r0 = np.sqrt((y_0-yc)**2 + (z_0-zc)**2)
                    theta0 = np.arctan2(z_0-zc,y_0-yc)
                    omega_scale = (1.0 -(x_0-x_min)/(x_max-x_min))
                    omega_scale = np.clip(omega_scale, 0.0, 1.0)
                    rotating_force.addParticle(atom_id, [r0, theta0, x_0, omega_scale])  
   

    ####################################################################
    system.addForce(rotating_force)  
    for atom_id in indices_edge:
        x0, y0, z0 = coords[atom_id]
        force_edge.addParticle(atom_id, [x0, y0, z0])
    system.addForce(force_edge)


    ##what gets saved from simulation runs
    dcdReporter = DCDReporter(system_name+'_plus2xspeed_flow.dcd', 25000)    # 1 PER 50 PS
    dataReporter = StateDataReporter(system_name+'_plus2xspeed_flow.log',
                                     5000,
                                     totalSteps=n_steps,
                                     step=True,
                                     speed=True,
                                     progress=True,
                                     elapsedTime=True,
                                     remainingTime=True,
                                     potentialEnergy=True,
                                     temperature=True,
                                     volume=True,
                                     density=True,
                                     separator=' ')
    checkpointReporter = CheckpointReporter(system_name+'_plus2xspeed_flow.chk', 5000)
    restartReporter = RestartReporter(system_name+'_plus2xspeed_flow.rst7',reportInterval=5000,netcdf=True)
                                     
    print('Creating Simulation object...')
    simulation = Simulation(topology, system, integrator, platform, platformProperties)  # GPU code
    simulation.context.setPositions(positions)  # specifies initial atom positions
    if parm.box_vectors is not None:  ##None is the singeton of Nonetype in python refers to the presence/absence of a value. S                                            o (is ==) and (is not !=)
        simulation.context.setPeriodicBoxVectors(*parm.box_vectors)
    # simulation.context.setParameter('k', value_of_k)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(dcdReporter)
    simulation.reporters.append(dataReporter)
    simulation.reporters.append(checkpointReporter)
    simulation.reporters.append(restartReporter)

    #### You can adjust this block to match the rotation speed you want ######
    dt_ps = dt.value_in_unit(picoseconds)    # 2 fs
    total_time_ps = n_steps * dt_ps
    omega = 4*np.pi/ total_time_ps    #change the sign of the omega to change the direction of the veoricity 
    time_ps = 0.0

    for step in range(n_steps):
        time_ps = step * dt_ps
        dphi = (omega * time_ps)        ####% (2*np.pi) is removed to skip any weird jumps due to the omega_scal
        simulation.context.setParameter("dphi", dphi)
        simulation.step(1)

        
    print('Done flow')
    

    print('Saving final state')
    final_state = simulation.context.getState(getForces=True, getVelocities=True, getPositions=True, getEnergy=True, getParameterDerivatives=True,  getParameters=True)
    outfile = open(system_name+'plus2xspeed_flow_DONE.xml', 'w+')
    outfile.write(XmlSerializer.serialize(final_state))
    outfile.close()

    
def main():
    parm_name = "/mnt/rahi_hdisk/chiral_flow/20mer/imp_4x20mer.prmtop"
    system_name = "/mnt/rahi_hdisk/chiral_flow/20mer/imp_4x20mer"
    flow(parm_name, system_name)


if __name__ == "__main__":
    main()
