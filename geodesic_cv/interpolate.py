import numpy as np
import torch
from tqdm import tqdm

from geodesic_cv.pointcloud import PointCloud


def uniform_interp_t(
    num_interp: int = 500,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    t_interp = torch.rand(num_interp, generator=generator, device=device)
    return t_interp


def gaussian_interp_t(
    num_interp: int = 500,
    mean: float = 0.5,
    std: float = 0.05,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    t_interp = torch.normal(
        mean, std, (num_interp,), generator=generator, device=device
    )
    t_interp = torch.clamp(t_interp, 0.0, 1.0)
    return t_interp


def geodesic_interpolation(
    xyz_unfolded: torch.Tensor,  # [n_frame, n_point, dim]
    xyz_folded: torch.Tensor,  # [n_frame, n_point, dim]
    t_interp: torch.Tensor,  # [n_interp]
    alpha: float = 1.0,
    seed: int = 42,
    use_tqdm: bool = True,
) -> torch.Tensor:
    assert xyz_unfolded.shape == xyz_folded.shape
    n_frame, n_point, dim = xyz_unfolded.shape
    rng = np.random.default_rng(seed)

    manifold = PointCloud(dim=dim, numpoints=n_point, base=xyz_folded[0], alpha=alpha)
    xyz_unfolded = manifold.align_mpoint(xyz_unfolded[None]).squeeze(0)
    xyz_folded = manifold.align_mpoint(xyz_folded[None]).squeeze(0)

    num_interp = t_interp.shape[0]
    unfolded_idx = rng.choice(n_frame, size=num_interp, replace=False)
    folded_idx = rng.choice(n_frame, size=num_interp, replace=False)

    xyz_interp_all = []

    iterator = tqdm(range(num_interp)) if use_tqdm else range(num_interp)
    for idx in iterator:
        xyz_interp = manifold.s_geodesic(
            xyz_unfolded[None, None, unfolded_idx[idx]],  # [1, 1, n_point, dim]
            xyz_folded[None, None, folded_idx[idx]],  # [1, 1, n_point, dim]
            t_interp[idx].unsqueeze(0),  # [1,]
        ).squeeze()  # [n_point, dim]
        xyz_interp_all.append(xyz_interp)

    xyz_interp_all = torch.stack(xyz_interp_all)  # [n_interp, n_point, dim]
    return xyz_interp_all  # [n_interp, n_point, dim]
