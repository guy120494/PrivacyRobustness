import argparse
import sys
from pathlib import Path

import kornia.metrics as metrics
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from CreateModel import create_model
from GetParams import str2list
from common_utils.common import load_weights
from evaluations import transform_vmin_vmax_batch
from utils import get_margin, get_distances_from_margin


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


def get_evaluation_score_dssim(xxx, yyy, ds_mean):
    xx = xxx.clone()
    yy = yyy.clone()

    # Scale to images
    yy += ds_mean
    xx = transform_vmin_vmax_batch(xx + ds_mean)

    # Score
    ssims = pairwise_ssim(xx, yy)
    dssim = (1 - ssims) / 2

    return dssim


def get_total_successful_reconstructions(path_to_reconstructions_folder: Path, path_to_training_images_file: Path,
                                         threshold: float = 0.4, device='cuda:0') -> (int, int):
    training_images = torch.load(str(path_to_training_images_file))['x'].to(device)
    total_of_successful_reconstructions = 0
    number_of_attacks = 0
    for file_path in path_to_reconstructions_folder.rglob('**/*x_final.pt*'):
        number_of_attacks += 1
        reconstructed_images = TensorDataset(torch.load(str(file_path)).to(device))
        reconstructed_images = DataLoader(
            reconstructed_images,
            batch_size=10,
            shuffle=False,  # Set to True for training
            drop_last=False
        )
        for i, batch_data in enumerate(reconstructed_images):
            current_batch = batch_data[0]
            dssim_success_matrix = get_evaluation_score_dssim(current_batch, training_images, ds_mean=0) < threshold
            total_of_successful_reconstructions += (dssim_success_matrix.sum(dim=0) > 0).sum().detach().item()
            del dssim_success_matrix
    return total_of_successful_reconstructions, number_of_attacks


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
    parser.add_argument('--model', type=str, default='')
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
    if not isinstance(args, list):
        args = args[0]
    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    path_to_reconstructions_folder = Path(args.reconstruction_folder)
    path_to_training_images_file = Path(args.train_file)
    print(f"RECONSTRUCTION FOLDER {path_to_reconstructions_folder}")
    print(f"TRAINING IMAGES {path_to_training_images_file}")
    print(get_total_successful_reconstructions(path_to_reconstructions_folder, path_to_training_images_file))

    torch.set_default_dtype(torch.float64)

    model = create_model(args, extraction=True)
    model.eval()
    model = load_weights(model, args.model)
    training_data = torch.load(str(path_to_training_images_file))
    loader = TensorDataset(training_data['x'].to(args.device), training_data['y'].to(args.device))
    loader = DataLoader(loader, batch_size=500, shuffle=False, drop_last=False)
    margin = get_margin(args, model, loader)
    distances = get_distances_from_margin(args, margin, model, loader)
    distances += margin

    k = 10
    bins = margin * (1 + 0.1 * np.arange(k + 1))
    hist, bins = np.histogram(distances.detach().cpu().numpy(), bins=bins)
    print("hist:", hist)
    print("bins:", bins)
