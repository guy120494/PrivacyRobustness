import argparse
import sys
from pathlib import Path

import kornia.metrics as metrics
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from CreateData import setup_problem
from CreateModel import create_model
from GetParams import str2list
from Main import setup_args
from common_utils.common import load_weights
from evaluations import transform_vmin_vmax_batch
from settings import results_base_dir
from utils import get_margin, get_distances_from_margin

import torch


def same_images_ignore_order(X: torch.Tensor, Y: torch.Tensor) -> bool:
    if X.shape != Y.shape:
        return False

    B = X.shape[0]

    X_flat = X.view(B, -1)
    Y_flat = Y.view(B, -1)

    X_sorted, _ = torch.sort(X_flat, dim=0)
    Y_sorted, _ = torch.sort(Y_flat, dim=0)
    return torch.equal(X_sorted, Y_sorted)


def deduplicate_images(x: torch.Tensor):
    """
    x: Tensor of shape (B, C, H, W)
    Returns:
        unique_images: Tensor of shape (B_unique, C, H, W)
        inverse_indices: Tensor mapping original indices -> unique indices
    """
    B = x.shape[0]
    x_flat = x.view(B, -1)  # (B, C*H*W)
    unique_flat, inverse = torch.unique(x_flat, dim=0, return_inverse=True)

    unique_images = unique_flat.view(-1, *x.shape[1:])
    return unique_images, inverse


def pairwise_ssim(imgs1, imgs2, window_size=3):
    B1, C, H, W = imgs1.shape
    B2 = imgs2.shape[0]

    # Expand for broadcasting: (B1, B2, C, H, W)
    i1 = imgs1[:, None].expand(B1, B2, C, H, W)
    i2 = imgs2[None, :].expand(B1, B2, C, H, W)

    # Flatten to batch: (B1*B2, C, H, W)
    i1 = i1.reshape(-1, C, H, W)
    i2 = i2.reshape(-1, C, H, W)

    # Compute SSIM per pair → (B1*B2,)
    ssim_vals = metrics.ssim(i1, i2, window_size=window_size).mean(dim=[1, 2, 3])

    # Reshape to pairwise matrix
    return ssim_vals.reshape(B1, B2)


def get_evaluation_score_dssim(xxx, yyy):
    """
    Get dssim matrix for xxx (reconstructions) and yyy (training images). Both should be with mean of training set added
     (yyy should be the real training images)
    @param xxx: reconstructions
    @param yyy: real training set images
    @return: DSSIM matrix for all dssim scores between xxx and yyy
    """
    xx = xxx.clone()
    yy = yyy.clone()

    # Scale reconstructions images
    xx = transform_vmin_vmax_batch(xx)

    # Score
    ssims = pairwise_ssim(xx, yy)
    dssim = (1 - ssims) / 2

    return dssim


def get_reconstructions_for_training_images(path_to_reconstructions_folder: Path, training_images, mean,
                                            device='cuda:0'):
    reconstructions = torch.zeros_like(training_images)
    best_dssim = torch.full((training_images.shape[0],), float('inf'), dtype=torch.float64).to(device)
    for file_path in path_to_reconstructions_folder.rglob('**/*x*.pt*'):
        reconstructed_images = get_reconstructed_images(file_path, device)
        for i, batch_data in enumerate(reconstructed_images):
            current_batch = batch_data[0] + mean
            dssim_matrix = get_evaluation_score_dssim(current_batch, training_images)
            col_min_val, col_min_idx = dssim_matrix.min(dim=0)
            mask = col_min_val < best_dssim
            best_dssim[mask] = col_min_val[mask]
            col_min_val = col_min_val.masked_fill(~mask, float('inf'))
            mask = torch.isfinite(col_min_val)
            reconstructions[mask] = current_batch[col_min_idx[mask]]

    return reconstructions


def get_successful_reconstructions(path_to_reconstructions_folder: Path, training_images,
                                   mean, threshold: float = 0.3, device='cuda:0'):
    successful_reconstructions = []
    reconstructed_training_images = []
    for file_path in path_to_reconstructions_folder.rglob('**/*x*.pt*'):
        reconstructed_images = get_reconstructed_images(file_path, device)
        for i, batch_data in enumerate(reconstructed_images):
            current_batch = batch_data[0] + mean
            dssim_matrix = get_evaluation_score_dssim(current_batch, training_images)
            row_min_vals, row_min_idx = dssim_matrix.min(dim=1)
            mask = row_min_vals < threshold
            successful_reconstructions.extend(torch.unbind(current_batch[mask]))
            reconstructed_training_images.extend(torch.unbind(training_images[row_min_idx[mask]]))

    return torch.stack(successful_reconstructions, dim=0), torch.stack(reconstructed_training_images, dim=0)


def get_total_number_of_successful_reconstructions(path_to_reconstructions_folder: Path, training_images, mean,
                                                   threshold: float = 0.3,
                                                   device='cuda:0') -> (int, int):
    successful_reconstructions = []
    number_of_attacks = 0
    for file_path in path_to_reconstructions_folder.rglob('**/*x*.pt*'):
        number_of_attacks += 1
        reconstructed_images = get_reconstructed_images(file_path, device)
        successful_batch = []
        for i, batch_data in enumerate(reconstructed_images):
            current_batch = batch_data[0]
            dssim_success_matrix = get_evaluation_score_dssim(current_batch + mean, training_images) < threshold
            successful_batch.append(dssim_success_matrix.any(dim=0))
            del dssim_success_matrix
        successful_batch = torch.stack(successful_batch, dim=0).any(dim=0)
        successful_reconstructions.append(successful_batch.clone())

    total_number_of_successful_reconstructions = torch.stack(successful_reconstructions, dim=0).any(
        dim=0).sum().detach().item()
    return total_number_of_successful_reconstructions, number_of_attacks


def get_reconstructed_images(file_path, device):
    reconstructed_images = TensorDataset(torch.load(str(file_path)).to(device))
    reconstructed_images = DataLoader(
        reconstructed_images,
        batch_size=10,
        shuffle=False,  # Set to True for training
        drop_last=False
    )
    return reconstructed_images


def print_histogram(data, bins=10, width=50, header=None):
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = hist.max()

    print("\nHistogram:") if header is None else print(f"\n {header} histogram")
    for left, right, count in zip(bin_edges[:-1], bin_edges[1:], hist):
        bar = "#" * int(width * (count / max_count))
        print(f"{left:8.2f} – {right:8.2f} | {bar} {count}")


def print_graph(points, width=60, height=20):
    """
    Print an ASCII graph of (x, y) points to stdout.
    """
    if not points:
        print("(no points)")
        return

    xs, ys = zip(*points)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Avoid division by zero
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1

    # Create empty canvas
    canvas = [[' ' for _ in range(width)] for _ in range(height)]

    def scale_x(x):
        return int((x - xmin) / (xmax - xmin) * (width - 1))

    def scale_y(y):
        return int((y - ymin) / (ymax - ymin) * (height - 1))

    # Plot points
    for x, y in points:
        cx = scale_x(x)
        cy = height - 1 - scale_y(y)
        canvas[cy][cx] = '*'

    # Draw axes (if zero is in range)
    if xmin <= 0 <= xmax:
        x0 = scale_x(0)
        for r in range(height):
            canvas[r][x0] = '|'

    if ymin <= 0 <= ymax:
        y0 = height - 1 - scale_y(0)
        for c in range(width):
            canvas[y0][c] = '-'

    if xmin <= 0 <= xmax and ymin <= 0 <= ymax:
        canvas[y0][x0] = '+'

    # Print
    for row in canvas:
        print(''.join(row))

    print(f"x ∈ [{xmin:.2f}, {xmax:.2f}], y ∈ [{ymin:.2f}, {ymax:.2f}]")


def generate_random_images():
    dir_path = Path("random_images")
    dir_path.mkdir(exist_ok=True)
    for i in range(10):
        imgs = torch.rand(10, 3, 64, 64)
        file_path = dir_path / f"images_{i:03d}.pt"
        torch.save(imgs, file_path)
        print(f"Saved {file_path} with shape {imgs.shape}")


def get_args(*args):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--reconstruction_folder', type=str)
    parser.add_argument('--first_reconstruction_folder', type=str)
    parser.add_argument('--second_reconstruction_folder', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--run_mode', type=str, default='reconstruct')
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--proj_name', type=str, default='')
    parser.add_argument('--problem', type=str, default='cifar10_vehicles_animals')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_dim', type=int, default=32 * 32 * 3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--model_use_bias', type=bool, default=True)
    parser.add_argument('--model_hidden_list', default='[1000, 1000]', type=str2list)
    parser.add_argument('--extraction_model_activation', type=str, default='modifiedrelu')
    parser.add_argument('--extraction_model_relu_alpha', default=300, type=float)
    parser.add_argument('--model_type', default='mlp', help='options: mlp')
    parser.add_argument('--use_init_scale', default=False, type=bool, help='')
    parser.add_argument('--data_reduce_mean', default=False, type=bool, help='')
    parser.add_argument('--wandb_active', default=False, type=bool, help='')
    parser.add_argument('--analyze_multiple_thresholds', default=True, type=bool, help='')
    parser.add_argument('--data_per_class_train', default=250, type=int, help='')
    parser.add_argument('--seed', default=1, type=int, help='')
    if not isinstance(args, list):
        args = args[0]
    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    path_to_reconstructions_folder = Path(args.reconstruction_folder)
    path_to_training_images_file = Path(args.train_file)
    training_images = torch.load(str(path_to_training_images_file))['x'].to(args.device)
    mean = training_images.mean(dim=[0, -2, -1]).detach()
    mean = mean.view(1, 3, 1, 1)

    print(f"DSSIM THRESHOLD {args.threshold}")
    print(f"RECONSTRUCTION FOLDER {path_to_reconstructions_folder}")
    print(f"TRAINING IMAGES {path_to_training_images_file}")

    if args.analyze_multiple_thresholds:
        thresholds = [0.2 + i * 0.05 for i in range(5)]
        points = []
        for threshold in thresholds:
            y = get_total_number_of_successful_reconstructions(path_to_reconstructions_folder, training_images, mean,
                                                               threshold=threshold)[0]
            points.append((threshold, y))
        print([p[0] for p in points])
        print([p[1] for p in points])

    if args.first_reconstruction_folder is not None and args.second_reconstruction_folder is not None:
        path_to_first_reconstruction_folder = Path(args.first_reconstruction_folder)
        path_to_second_reconstruction_folder = Path(args.second_reconstruction_folder)
        _, t1 = get_successful_reconstructions(path_to_first_reconstruction_folder, training_images, mean,
                                               args.threshold)
        # t1 = deduplicate_images(t1)[0]
        y1 = get_reconstructions_for_training_images(path_to_first_reconstruction_folder, t1, mean)
        y2 = get_reconstructions_for_training_images(path_to_second_reconstruction_folder, t1, mean)

        save_path = Path(results_base_dir) / args.save_folder / "compare_models.pth"
        save_path.mkdir(exist_ok=True)
        torch.save(
            {"reconstructed_training": t1, "first_model_reconstructions": y1, "second_model_reconstructions": y2},
            save_path)

    # torch.set_default_dtype(torch.float64)
    # model = create_model(args, extraction=True)
    # model.eval()
    # model = load_weights(model, args.model)
    # training_data = torch.load(str(path_to_training_images_file))
    # loader = TensorDataset(training_data['x'].to(args.device), training_data['y'].to(args.device))
    # loader = DataLoader(loader, batch_size=500, shuffle=False, drop_last=False)
    # margin = get_margin(args, model, loader)
    # distances = get_distances_from_margin(args, margin, model, loader)
    #
    # k = 10
    # bins = margin * (1 + 0.1 * np.arange(k + 1))
    # print_histogram(data=(margin + distances).detach().cpu().numpy(), bins=bins, header="margin")
    # print_histogram(data=distances.detach().cpu().numpy(), header="distance form margin")
