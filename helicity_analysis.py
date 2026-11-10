import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md
from cycler import cycler
from scipy.stats import linregress



def get_monomer_atom_indices(topology):     #returns a [[(array)]_i]_j;  j = id of the pol i = id of the mon inside the pol : array contains the array of atom ids of the resid_ij 
    monomer_indices = []
    for polymer_resids in POLYMER_RESIDS:
        polymer_indices = []
        for resid in polymer_resids:
            residue = topology.residue(resid)
            heavy_atoms = [atom.index for atom in residue.atoms
                    if atom.element is not None and atom.element.symbol != "H"]
            polymer_indices.append(np.asarray(heavy_atoms, dtype=int))
        monomer_indices.append(polymer_indices)
    return monomer_indices


def calculate_all_cogs(traj, frame):            # Function to calculate the monomer's center of geomtery 
    cogs = np.zeros((N_POLYMERS, N_MONOMERS, 3))
    xyz = traj.xyz[frame]
    for p in range(N_POLYMERS):
        for i in range(N_MONOMERS):
            atom_indices = MONOMER_ATOM_INDICES[p][i]
            cogs[p, i] = np.mean(xyz[atom_indices], axis=0)
    return cogs


def calculate_moving_centerline(cogs):          # calculates the cog of the cog of all the 4 monomers 
    return np.mean(cogs, axis=0)                # Shape: (20, 3)



def calculate_theta_raw(cogs, centerline):      # Function to calculate the raw theta and also the distance for the centerline
    dy = (cogs[:, :, 1] - centerline[None, :, 1])
    dz = (cogs[:, :, 2] - centerline[None, :, 2])
    theta_raw = np.arctan2(dz, dy)          # gets the raw angles 
    radius = np.hypot(dy, dz)               # gets the radial distance of the monomer from the moving center-line
    return theta_raw, radius


def fit_helicity(x, theta):                     # function to do the dtheta/dx fit to get the k 

    result = linregress(x, theta)
    k = result.slope
    r2 = result.rvalue**2
    pitch = np.nan
    if abs(k) > K_THRESHOLD:
        pitch = 2 * np.pi / abs(k)
    return k, pitch, r2



# =====================Input Parameters=========================================

TRAJ_FILE = "/mnt/rahi_hdisk/chiral_flow/20mer/imp_4x20mer_minus2xspeed_flow.dcd"
TOP_FILE = "/mnt/rahi_hdisk/chiral_flow/20mer/imp_4x20mer.prmtop"

FRAME_INTERVAL_NS = 0.05      #frame save rate 0.05

N_POLYMERS = 4
N_MONOMERS = 20

K_THRESHOLD = 1e-6

R_CUTOFF = 0.05   # cutoff distance you can allow a monomer to approach the centerline to be considered for the k-fitting 
MIN_FIT_MONOMERS = 5    # Minimum number of consecutive valid monomers required for a helicity fit.

POLYMER_RESIDS = [
    list(range(0, 20)),
    list(range(39, 19, -1)),
    list(range(59, 39, -1)),
    list(range(60, 80))
]

traj = md.load(TRAJ_FILE, top=TOP_FILE)
N_FRAMES = traj.n_frames
time = np.arange(N_FRAMES) * FRAME_INTERVAL_NS

MONOMER_ATOM_INDICES = get_monomer_atom_indices(traj.topology)

###--------First initiate the required arrays with zeros 
cogs_all = np.zeros((N_FRAMES, N_POLYMERS, N_MONOMERS, 3))
centerlines_all = np.zeros((N_FRAMES, N_MONOMERS, 3))
theta_raw_all = np.zeros((N_FRAMES, N_POLYMERS, N_MONOMERS))
radius_all = np.zeros_like(theta_raw_all)       # creates a 0 array same shape ad theta_raw_all

for frame in range(N_FRAMES):
    cogs = calculate_all_cogs(traj, frame)
    centerline = calculate_moving_centerline(cogs)
    theta_raw, radius = calculate_theta_raw(cogs, centerline)
    cogs_all[frame] = cogs
    centerlines_all[frame] = centerline
    theta_raw_all[frame] = theta_raw
    radius_all[frame] = radius
    
# creating a distance based mask that will be exclude monomers too close to the centre-line during the fitting 
valid_mask = (np.isfinite(radius_all) & (radius_all >= R_CUTOFF))
n_included = np.sum(valid_mask, axis=2)


#-------------performing temporal unwrap to avoid discontinuities from the arctan2 function ------------------#
delta_raw = np.diff(theta_raw_all, axis=2)      # the raw diff vals are calculated along the axis of the 20 monomers, so results in 19 along this axis 
delta_wrapped = np.arctan2(np.sin(delta_raw),np.cos(delta_raw))        # trick to wrap these delta values [-pi, pi]
pair_valid = (valid_mask[:, :, :-1] & valid_mask[:, :, 1:]) # consecutive monomer pair is valid only if both the monomers are not v close to the centreline
delta_wrapped[~pair_valid] = np.nan     # the indices belonging to monomer pairs which are not valid is replaced my nan

delta_temporal = np.full_like(delta_wrapped, np.nan)    # Initialize array to store temporally unwrapped differences
for p in range(N_POLYMERS):
    for pair in range(N_MONOMERS - 1):
        valid_frames = np.flatnonzero(pair_valid[:, p, pair])   # Find frames where this monomer pair is valid
        if len(valid_frames) == 0:
            continue
        # Identify gaps in the valid frames
        splits = np.where(np.diff(valid_frames) > 1)[0] + 1
        # Split into consecutive valid-frame intervals
        segments = np.split(valid_frames, splits)
        # Temporally unwrap each interval independently
        for frames in segments:
            delta_temporal[frames, p, pair] = np.unwrap(delta_wrapped[frames, p, pair])




#---------- reconstrcuting the angular positions using the temporally unwrapped deltas 
theta_temporal = np.full_like(theta_raw_all, np.nan)
for frame in range(N_FRAMES):
    for p in range(N_POLYMERS):
        valid_indices = np.flatnonzero(valid_mask[frame, p])
        if len(valid_indices) == 0:
            continue
        splits = np.where(np.diff(valid_indices) > 1)[0] + 1
        # Separate into continuous valid segments
        segments = np.split(valid_indices, splits)
        for indices in segments:
            start = indices[0]
            # Reference angle for this segment
            theta_temporal[frame, p, start] = (theta_raw_all[frame, p, start])
            # Reconstruct angles within the segment
            for i in indices[:-1]:
                theta_temporal[frame, p, i + 1] = (theta_temporal[frame, p, i] + delta_temporal[frame, p, i])

# this block adjust the dicontinutites btw the valid segmntbs                
for frame in range(N_FRAMES):
    for p in range(N_POLYMERS):
        valid_indices = np.flatnonzero(valid_mask[frame, p])
        if len(valid_indices) < 2:
            continue
        splits = np.where(np.diff(valid_indices) > 1)[0] + 1
        segments = np.split(valid_indices, splits)
        for s in range(1, len(segments)):
            previous_end = segments[s - 1][-1]
            current_start = segments[s][0]
            difference = (theta_temporal[frame, p, current_start] - theta_temporal[frame, p, previous_end])
            # Find the integer multiple of 2π (360°) needed to bring the angular
            # difference between two disconnected segments into [-π, π].
            # If the difference is already within this range, shift = 0.
            # This aligns the segments using the shortest angular difference across the gap
            shift = -2 * np.pi * np.round(difference / (2 * np.pi))     #                                        
            theta_temporal[frame, p, segments[s]] += shift


k_temporal = np.full((N_FRAMES, N_POLYMERS), np.nan)
pitch_temporal = np.full((N_FRAMES, N_POLYMERS), np.nan)
r2_temporal = np.full((N_FRAMES, N_POLYMERS), np.nan)

for frame in range(N_FRAMES):
    for p in range(N_POLYMERS):
        x = cogs_all[frame, p, :, 0]    # x coords of all the monomers 
        theta = theta_temporal[frame, p, :]
        radius = radius_all[frame, p, :]
        mask = (valid_mask[frame, p] & np.isfinite(x) & np.isfinite(theta))
        if np.sum(mask) < MIN_FIT_MONOMERS:
            continue
        result = linregress(x[mask], theta[mask])   #fit the using the mons that agree with all the conditons 
        k = result.slope
        r2 = result.rvalue**2
        
        k_temporal[frame, p] = k
        r2_temporal[frame, p] = r2
        pitch_temporal[frame, p] = (2*np.pi / abs(k))

# ============================================================
# FIT HELICITY USING ONLY MONOMERS ABOVE RADIAL CUTOFF
# ============================================================

k_temporal = np.full((N_FRAMES, N_POLYMERS), np.nan)
pitch_temporal = np.full((N_FRAMES, N_POLYMERS), np.nan)
r2_temporal = np.full((N_FRAMES, N_POLYMERS), np.nan)

for frame in range(N_FRAMES):

    for p in range(N_POLYMERS):
        x = cogs_all[frame, p, :, 0]
        theta = theta_temporal[frame, p, :]

        # Apply radial cutoff ONLY during regression
        mask = (
            valid_mask[frame, p]
            & np.isfinite(x)
            & np.isfinite(theta)
        )

        if np.sum(mask) < MIN_FIT_MONOMERS:
            continue

        # if np.ptp(x[mask]) < 1e-10:
        #     continue

        result = linregress(x[mask], theta[mask])

        k = result.slope

        k_temporal[frame, p] = k
        r2_temporal[frame, p] = result.rvalue**2

        if np.isfinite(k) and abs(k) > K_THRESHOLD:
            pitch_temporal[frame, p] = 2 * np.pi / abs(k)


# saving the temporal k as a df      
df_k = pd.DataFrame(k_temporal, columns=[f"Poly_{p+1}" for p  in range(N_POLYMERS)])  
df_k.insert(0,"Time_ns",time)
df_k.to_csv(f"k_temporal_minus2x.csv", index=False)
        
fig, axes = plt.subplots(figsize=(12, 6))
colors = ["#FF0080", "#3d2fa5", "#19B357", "#f17918"]
axes.set_prop_cycle(cycler(color=colors))
for p in range(N_POLYMERS):
    axes.plot(
        time,
        k_temporal[:, p],
        label=f"Polymer {p+1}",
        alpha=0.85,
        linewidth=1.5
    )
axes.set_xlabel("Time (ns)", fontsize=18, fontweight="bold")
axes.set_ylabel("k (rad/nm)", fontsize=18, fontweight="bold")
axes.tick_params(axis="both", labelsize=15, length=6,width=1.5,direction="in")
axes.axhline(0,color="gray", linestyle="--",linewidth=1.0)
axes.legend(loc="upper right", bbox_to_anchor=(0.15, 0.99), fontsize=12,
            prop={"size": 12, "weight": "bold"}, handlelength=1.2)
for spine in axes.spines.values():
    spine.set_linewidth(1.8)
    spine.set_color("black")
plt.tight_layout()
plt.savefig("k_temporal_minus2x.png", dpi=300, bbox_inches="tight")
plt.show()


