# MBAR for eABF simulations from https://github.com/ochsenfeld-lab/adaptive_sampling
# Andreas Hulm, Johannes C. B. Dietschreit, and Christian Ochsenfeld,
# "Statistically optimal analysis of the extended-system adaptive biasing force (eABF) method",
# J. Chem. Phys. 157, 024110 (2022) https://doi.org/10.1063/5.0095554

import argparse
from pathlib import Path

import numpy as np
from adaptive_sampling.processing_tools import mbar
from adaptive_sampling.processing_tools.utils import DeltaF_fromweights
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--colvar-file", type=Path, required=True)
parser.add_argument("--cv-thresh", type=float, nargs=2, required=True)
parser.add_argument("--sigma", type=float, required=True)
parser.add_argument("--save-path", type=Path, required=True)
parser.add_argument("--equil-temp", type=float, default=340)
parser.add_argument("--skip-steps", type=int, default=0)
parser.add_argument("--unit-steps", type=int, default=25000)
parser.add_argument("--ns-per-step", type=float, default=0.002)
args = parser.parse_args()

# Save path
args.save_path.mkdir(exist_ok=True, parents=True)

# Load the normal and extended system trajectory (COLVAR file)
traj_dat = np.genfromtxt(args.colvar_file, skip_header=1)
cv = traj_dat[:, 1]
lambd = traj_dat[:, 2]
total_steps = len(traj_dat)

cv_thresh = [
    args.cv_thresh[0],
    (args.cv_thresh[0] + args.cv_thresh[1]) / 2,  # approximate TS
    args.cv_thresh[1],
]

# Time points to compute the PMF
step_grid = np.arange(
    args.skip_steps + args.unit_steps, total_steps + 1, args.unit_steps
)

# MBAR runs
Delta_Fs = []
for current_step in tqdm(step_grid):
    traj_cv = cv[args.skip_steps : current_step]
    traj_lam = lambd[args.skip_steps : current_step]
    grid = np.arange(traj_cv.min(), traj_cv.max() + args.sigma / 2, args.sigma)

    # Split the eABF frames into virtual umbrella windows
    traj_list, indices, meta_f = mbar.get_windows(
        grid, traj_cv, traj_lam, args.sigma, equil_temp=args.equil_temp
    )

    # Compute the Boltzmann factors
    exp_U, frames_per_traj = mbar.build_boltzmann(
        traj_list, meta_f, equil_temp=args.equil_temp
    )

    # Compute the weights
    W = mbar.run_mbar(
        exp_U, frames_per_traj, outfreq=1000, conv=1.0e-6, max_iter=100000
    )

    Delta_F = DeltaF_fromweights(
        xi_traj=traj_cv[indices],
        weights=W,
        cv_thresh=cv_thresh,
        T=args.equil_temp,
    )
    Delta_Fs.append(Delta_F)

# Save the Delta_Fs
time_points = step_grid * args.ns_per_step
Delta_Fs = np.array(Delta_Fs)
np.savetxt(args.save_path / "Delta_Fs.log", [time_points, Delta_Fs], fmt="%.6f")

# Compute and save the PMF
pmf, _ = mbar.pmf_from_weights(grid, traj_cv[indices], W, equil_temp=args.equil_temp)
pmf -= pmf.min()
np.savetxt(args.save_path / "pmf.log", [grid, pmf], fmt="%.6f")
