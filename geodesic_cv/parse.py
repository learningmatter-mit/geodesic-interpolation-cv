from pathlib import Path

import mdtraj as md
import torch


def load_pdb(
    pdb_file: Path,
    heavy_only: bool = True,
) -> md.Trajectory:
    frame = md.load(pdb_file).center_coordinates()
    if heavy_only:
        atom_slice = frame.top.select("not element H")
        frame = frame.atom_slice(atom_slice)
    return frame


def load_xtc(
    xtc_file: Path,
    pdb_folded: Path = Path("data/5AWL_gro.pdb"),
    heavy_only: bool = True,
    stride: int = 1,
) -> md.Trajectory:
    frame_folded = md.load(pdb_folded).center_coordinates()
    top = frame_folded.top
    try:
        traj = md.load_xtc(xtc_file, top=top, stride=stride)
    except ValueError:
        # Try loading heavy atoms only\
        if heavy_only:
            frame_folded = frame_folded.atom_slice(top.select("not element H"))
            traj = md.load_xtc(xtc_file, top=frame_folded.top)
            traj = traj.center_coordinates().superpose(frame_folded[0])
            return traj
        else:
            raise ValueError("Failed to load xtc file")
    if heavy_only:
        atom_slice = top.select("not element H")
        frame_folded = frame_folded.atom_slice(atom_slice)
        traj = traj.atom_slice(atom_slice)
    traj = traj.center_coordinates().superpose(frame_folded[0])
    return traj


def create_contact_features(
    traj: md.Trajectory,
    R_0: float = 0.8,
    D_0: float = 0.0,
    n: int = 6,
    m: int = 12,
    eps: float = 1e-8,
) -> torch.Tensor:
    ca_pos = torch.tensor(traj.atom_slice(traj.top.select("name CA")).xyz)
    ca_dist = torch.cdist(ca_pos, ca_pos)
    triu_idx = torch.triu_indices(ca_dist.size(1), ca_dist.size(1), offset=1)
    ca_dist = ca_dist[:, triu_idx[0], triu_idx[1]]
    feat_d = (ca_dist - D_0) / R_0
    feat = (1.0 - feat_d**n) / (1.0 - feat_d**m + eps)
    return feat


def create_distance_features(
    traj: md.Trajectory,
) -> torch.Tensor:
    ca_pos = torch.tensor(traj.atom_slice(traj.top.select("name CA")).xyz)
    ca_dist = torch.cdist(ca_pos, ca_pos)
    triu_idx = torch.triu_indices(ca_dist.size(1), ca_dist.size(1), offset=1)
    ca_dist = ca_dist[:, triu_idx[0], triu_idx[1]]
    feat = ca_dist[:, ca_dist[0] > 0.0]
    return feat
