import argparse
from pathlib import Path

import mdtraj as md
import torch
from geodesic_cv.interpolate import (
    gaussian_interp_t,
    geodesic_interpolation,
    uniform_interp_t,
)
from geodesic_cv.parse import load_xtc

parser = argparse.ArgumentParser()
parser.add_argument("--xtc-unfolded", type=Path, required=True)
parser.add_argument("--xtc-folded", type=Path, required=True)
parser.add_argument("--num-interp", type=int, default=5000)
parser.add_argument("--traj-stride", type=int, default=10)
parser.add_argument("--pdb-folded", type=Path, default=Path("data/5AWL_gro.pdb"))
parser.add_argument(
    "--interp-method", type=str, default="gaussian", choices=["uniform", "gaussian"]
)
parser.add_argument("--gaussian-mean", type=float, default=0.5)
parser.add_argument("--gaussian-std", type=float, default=0.05)
parser.add_argument("--save-path", type=Path, default=Path("simulations/interpolation"))
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


# Save path
args.save_path.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Trajectory
traj_unfolded = load_xtc(args.xtc_unfolded, pdb_folded=args.pdb_folded)
traj_folded = load_xtc(args.xtc_folded, pdb_folded=args.pdb_folded)
xyz_unfolded = torch.tensor(traj_unfolded.xyz)[:: args.traj_stride]
xyz_folded = torch.tensor(traj_folded.xyz)[:: args.traj_stride]

# Interpolation
if args.interp_method == "gaussian":
    interp_t = gaussian_interp_t(
        num_interp=args.num_interp,
        mean=args.gaussian_mean,
        std=args.gaussian_std,
        seed=args.seed,
        device=device,
    )
else:
    interp_t = uniform_interp_t(
        num_interp=args.num_interp, seed=args.seed, device=device
    )
xyz_inter = geodesic_interpolation(xyz_unfolded, xyz_folded, interp_t, use_tqdm=True)

# Save
traj_inter = md.Trajectory(xyz=xyz_inter.cpu().numpy(), topology=traj_unfolded.top)
traj_inter.save_xtc(args.save_path / "trajout.xtc")
torch.save(interp_t.cpu(), args.save_path / "interp_t.pt")
