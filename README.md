# Exceptional homochirality and spin absorption via meniscus-driven chiral symmetry breaking

This repository contains the Python code used for MD simulations and the trajectory analysis used to generate the figures in this mamuscript 

## MD simulation data 
Simulations were conducted for an aggregate of 4, PII-2T polymers. The MD trajectory files for the vorticity induced MD simulations and the no force simulations can be found at https://uofi.app.box.com/folder/436277622435

## Vorticity induced MD Simulations 
vorticity_grad_MD.py : OpenMM script for simulating the rotational motion of polymer chains under an imposed vorticity gradient.

## Effective helical slope analysis 
* helicity_analysis.py was used to calculate the effective helical slope of individyal polymer chains at each saved MD simulation trajectory frame
* effective_helical_slopes/ Directory containing the calculated effective helical slope data in CSV format
* helicity_plots.ipynb plotting code used to generate Figure 5c of the manuscript