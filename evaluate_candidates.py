#!/usr/bin/env python3
"""
Evaluate reconstruction candidates against training data.

Recursively scans <folder> for all subdirectories named 'x', collects every
.pth candidate file found in them, concatenates into one tensor, loads the
training set for the given problem, then computes pairwise DSSIM / LPIPS /
CLIP scores for every (training sample, candidate) pair.

Output:  (N_train, N_candidates) score matrix per metric.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from common_utils.image import get_ssim_all
from CreateData import setup_problem


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_candidates(folder: Path, final_only: bool = False) -> torch.Tensor:
    # Collect every directory named 'x' anywhere under folder
    x_dirs = sorted(p for p in folder.rglob('x') if p.is_dir())
    if not x_dirs:
        raise FileNotFoundError(f"No subdirectory named 'x' found under {folder}")

    # Gather .pth files across all x/ directories
    pattern = 'x_final.pth' if final_only else '*.pth'
    pth_files: list[Path] = []
    for x_dir in x_dirs:
        pth_files.extend(sorted(x_dir.glob(pattern)))

    if not pth_files:
        needle = 'x_final.pth' if final_only else '.pth files'
        raise FileNotFoundError(f"No {needle} found in any 'x' subdirectory under {folder}")

    print(f"Found {len(x_dirs)} 'x' subdirectory/ies, {len(pth_files)} candidate file(s) total")
    tensors = []
    for f in pth_files:
        t = torch.load(f, map_location='cpu')
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected a tensor in {f}, got {type(t)}")
        if t.dim() == 3:
            t = t.unsqueeze(0)
        tensors.append(t.detach())
        print(f"  {f.relative_to(folder)}: {tuple(t.shape)}")

    candidates = torch.cat(tensors, dim=0)
    print(f"Total candidates tensor: {tuple(candidates.shape)}")
    return candidates


def _make_problem_args(problem, data_per_class_train, datasets_dir, device):
    args = argparse.Namespace()
    args.problem = problem
    args.data_per_class_train = data_per_class_train
    args.data_per_class_val = 0
    args.data_per_class_test = 500
    args.data_reduce_mean = True
    args.run_mode = 'train'
    args.device = device
    args.datasets_dir = datasets_dir
    args.extraction_data_amount_per_class = data_per_class_train
    return args


def load_training_data(problem, data_per_class_train, datasets_dir, device):
    args = _make_problem_args(problem, data_per_class_train, datasets_dir, device)
    train_loader, _, _ = setup_problem(args)
    Xtrn, Ytrn = next(iter(train_loader))
    return Xtrn.to(device).float(), Ytrn


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------

def _ensure_rgb(t: torch.Tensor) -> torch.Tensor:
    """Repeat single-channel images to 3 channels for LPIPS / CLIP."""
    if t.shape[1] == 1:
        t = t.expand(-1, 3, -1, -1)
    return t


def _to_lpips_input(t: torch.Tensor) -> torch.Tensor:
    """Scale to [-1, 1] and upscale if smaller than 64px (LPIPS requirement)."""
    t = _ensure_rgb(t) * 2 - 1
    if t.shape[-1] < 64 or t.shape[-2] < 64:
        scale = 64 / min(t.shape[-2], t.shape[-1])
        t = F.interpolate(t, scale_factor=scale, mode='bicubic', align_corners=False)
    return t


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _to_clip_input(t: torch.Tensor, device) -> torch.Tensor:
    t = _ensure_rgb(t)
    t = F.interpolate(t, size=(224, 224), mode='bicubic', align_corners=False)
    return (t - _CLIP_MEAN.to(device)) / _CLIP_STD.to(device)


# ---------------------------------------------------------------------------
# Per-metric matrix computation
# ---------------------------------------------------------------------------

def compute_dssim_matrix(xxx: torch.Tensor, yyy: torch.Tensor,
                         ds_mean: torch.Tensor, cand_batch_size: int = 5000) -> torch.Tensor:
    """
    Returns (N_train, N_cands) DSSIM matrix.
    xxx is on CPU; candidate batches are streamed to GPU.
    get_ssim_all(x_batch, y) -> (batch, N_train); transposed to fill matrix columns.
    """
    device = yyy.device
    ds_mean_cpu = ds_mean.cpu()
    N_cands = xxx.shape[0]
    N_train = yyy.shape[0]

    x_vis_cpu = (xxx + ds_mean_cpu).clamp(0, 1)  # CPU
    y_vis = (yyy + ds_mean).clamp(0, 1)           # GPU

    mat = torch.zeros(N_train, N_cands)
    for j in range(0, N_cands, cand_batch_size):
        x_batch = x_vis_cpu[j:j + cand_batch_size].to(device)
        # get_ssim_all(x, y) -> (N_x, N_y); here (batch, N_train)
        ssim_batch = get_ssim_all(x_batch, y_vis)
        mat[:, j:j + cand_batch_size] = ((1.0 - ssim_batch) / 2.0).t().cpu()
        print(f"  DSSIM: {min(j + cand_batch_size, N_cands)}/{N_cands} candidates done")

    return mat


def compute_lpips_matrix(xxx: torch.Tensor, yyy: torch.Tensor,
                         ds_mean: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """
    Returns (N_train, N_cands) LPIPS matrix.
    xxx is on CPU; candidate batches are streamed to GPU per training sample.
    """
    import lpips as lpips_lib
    device = yyy.device
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    N_train = yyy.shape[0]
    N_cands = xxx.shape[0]

    ds_mean_cpu = ds_mean.cpu()
    x_vis_cpu = _to_lpips_input((xxx + ds_mean_cpu).clamp(0, 1))  # CPU
    y_vis = _to_lpips_input((yyy + ds_mean).clamp(0, 1))          # GPU

    mat = torch.zeros(N_train, N_cands)
    with torch.no_grad():
        for i in range(N_train):
            yi = y_vis[i:i + 1]  # (1, C, H, W) on GPU
            row = []
            for j in range(0, N_cands, batch_size):
                xb = x_vis_cpu[j:j + batch_size].to(device)
                s = lpips_fn(xb, yi.expand(xb.shape[0], -1, -1, -1)).squeeze(dim=-1).squeeze(dim=-1).squeeze(dim=-1)
                row.append(s.cpu())
            mat[i] = torch.cat(row)
            if (i + 1) % 50 == 0:
                print(f"  LPIPS: {i + 1}/{N_train} training samples done")

    return mat  # (N_train, N_cands)


def compute_clip_matrix(xxx: torch.Tensor, yyy: torch.Tensor,
                        ds_mean: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """
    Returns (N_train, N_cands) cosine-distance matrix via CLIP embeddings.
    xxx is on CPU; batches are streamed to GPU for encoding.
    Features are accumulated on CPU to avoid OOM.
    """
    import clip
    device = yyy.device
    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model.eval()

    ds_mean_cpu = ds_mean.cpu()
    x_vis_cpu = (xxx + ds_mean_cpu).clamp(0, 1)  # CPU
    y_vis = (yyy + ds_mean).clamp(0, 1)           # GPU

    def _encode_from_cpu(imgs_cpu: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(0, imgs_cpu.shape[0], batch_size):
            batch = imgs_cpu[i:i + batch_size].to(device)
            with torch.no_grad():
                feats.append(clip_model.encode_image(_to_clip_input(batch, device)).float().cpu())
        return F.normalize(torch.cat(feats, dim=0), dim=1)  # CPU

    def _encode_from_gpu(imgs_gpu: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(0, imgs_gpu.shape[0], batch_size):
            with torch.no_grad():
                feats.append(clip_model.encode_image(_to_clip_input(imgs_gpu[i:i + batch_size], device)).float().cpu())
        return F.normalize(torch.cat(feats, dim=0), dim=1)  # CPU

    print(f"  Encoding {xxx.shape[0]} candidates ...")
    x_feat = _encode_from_cpu(x_vis_cpu)  # (N_cands, D) on CPU
    print(f"  Encoding {yyy.shape[0]} training images ...")
    y_feat = _encode_from_gpu(y_vis)       # (N_train, D) on CPU

    sim_mat = y_feat @ x_feat.t()         # (N_train, N_cands) on CPU
    return 1.0 - sim_mat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

THRESHOLDS = {'dssim': 0.3, 'lpips': 0.5, 'clip': 0.3}
METRIC_LABELS = {'dssim': 'DSSIM < 0.30', 'lpips': 'LPIPS < 0.50', 'clip': 'CLIP dist < 0.30'}


def main():
    parser = argparse.ArgumentParser(
        description='Pairwise DSSIM / LPIPS / CLIP evaluation of reconstruction candidates',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--folder', required=True,
                        help='Root folder containing an x/ subfolder with .pth candidate files')
    parser.add_argument('--problem', required=True,
                        choices=['cifar10_vehicles_animals', 'mnist_odd_even',
                                 'celeba_male_female', 'imagenet', 'sphere'],
                        help='Dataset / problem name')
    parser.add_argument('--data_per_class_train', type=int, required=True,
                        help='Number of training samples per class used during the attack')
    parser.add_argument('--datasets_dir', default=None,
                        help='Path to the datasets root (defaults to settings.py)')
    parser.add_argument('--device', default=None,
                        help='Torch device, e.g. cuda or cpu (auto-detected if omitted)')
    parser.add_argument('--metrics', nargs='+', default=['dssim', 'lpips', 'clip'],
                        choices=['dssim', 'lpips', 'clip'],
                        help='Metrics to compute')
    parser.add_argument('--final_only', type=lambda v: v.lower() in ('1', 'true', 'yes'), default=False,
                        help='Load only x_final.pth from each x/ subdirectory instead of all .pth files')
    parser.add_argument('--save_matrices', action='store_true',
                        help='Save each (N_train, N_cands) matrix as a .pth file inside --folder')
    parser.add_argument('--lpips_batch_size', type=int, default=256,
                        help='Candidate batch size for LPIPS row computation')
    parser.add_argument('--cand_batch_size', type=int, default=5000,
                        help='Candidate batch size for DSSIM (controls GPU memory usage)')
    args = parser.parse_args()

    device = torch.device(args.device if args.device
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    if args.datasets_dir is None:
        from settings import datasets_dir
        args.datasets_dir = datasets_dir

    from settings import results_base_dir
    folder = Path(results_base_dir) / args.folder

    # ---- Load candidates ----
    print("\n=== Loading candidates ===")
    # Keep candidates on CPU — they may be very large (e.g. 66k × 3 × 64 × 64).
    # Each metric function streams batches to GPU as needed.
    xxx = load_candidates(folder, final_only=args.final_only).float()

    # ---- Load training data ----
    print("\n=== Loading training data ===")
    Xtrn, Ytrn = load_training_data(
        args.problem, args.data_per_class_train, args.datasets_dir, device
    )

    # Follow analysis_utils.py convention: per-pixel mean subtraction
    ds_mean = Xtrn.mean(dim=0, keepdim=True)
    yyy = Xtrn - ds_mean

    print(f"\nCandidates : {tuple(xxx.shape)}")
    print(f"Training   : {tuple(yyy.shape)}")
    print(f"ds_mean    : {tuple(ds_mean.shape)}")

    # ---- Compute metrics ----
    import time
    matrices: dict[str, torch.Tensor] = {}
    t_start = time.time()
    candidates_tag = 'final_only' if args.final_only else 'all'

    if 'dssim' in args.metrics:
        print("\n=== Computing DSSIM (all pairs) ===")
        t0 = time.time()
        mat = compute_dssim_matrix(xxx, yyy, ds_mean, cand_batch_size=args.cand_batch_size)
        print(f"  DSSIM took {time.time() - t0:.1f}s")
        matrices['dssim'] = mat
        if args.save_matrices:
            out = folder / f'eval_dssim_matrix_{candidates_tag}.pth'
            torch.save(mat, out)
            print(f"  Saved → {out}")

    if 'lpips' in args.metrics:
        print("\n=== Computing LPIPS (all pairs) ===")
        t0 = time.time()
        mat = compute_lpips_matrix(xxx, yyy, ds_mean, batch_size=args.lpips_batch_size)
        print(f"  LPIPS took {time.time() - t0:.1f}s")
        matrices['lpips'] = mat
        if args.save_matrices:
            out = folder / f'eval_lpips_matrix_{candidates_tag}.pth'
            torch.save(mat, out)
            print(f"  Saved → {out}")

    if 'clip' in args.metrics:
        print("\n=== Computing CLIP (all pairs) ===")
        t0 = time.time()
        mat = compute_clip_matrix(xxx, yyy, ds_mean)
        print(f"  CLIP took {time.time() - t0:.1f}s")
        matrices['clip'] = mat
        if args.save_matrices:
            out = folder / f'eval_clip_matrix_{candidates_tag}.pth'
            torch.save(mat, out)
            print(f"  Saved → {out}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"{'Metric':<8}  {'Matrix shape':<20}  {'Best-cand mean':>14}  {'Successful':>12}")
    print("-" * 60)
    N_train = yyy.shape[0]
    for name, mat in matrices.items():
        best = mat.min(dim=1).values          # best candidate per training sample
        n_ok = (best < THRESHOLDS[name]).sum().item()
        print(f"{name.upper():<8}  {str(tuple(mat.shape)):<20}  "
              f"{best.mean().item():>14.4f}  {n_ok:>5}/{N_train}  ({METRIC_LABELS[name]})")
    print("-" * 60)
    print(f"Total metrics time: {time.time() - t_start:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
