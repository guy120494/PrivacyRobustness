import argparse
import sys
from pathlib import Path

import kornia.metrics as metrics
import torch
from torch import Tensor

from evaluations import transform_vmin_vmax_batch


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


def get_dssim_matrix_for_all_attacks(path_to_reconstructions_folder: Path, path_to_training_images_file: Path,
                                     device='cuda:0') -> Tensor:
    training_images = torch.load(str(path_to_training_images_file)).to(device)
    final_matrix = []
    for file_path in path_to_reconstructions_folder.rglob('*x_final.pt'):
        reconstructed_images = torch.load(str(file_path)).to(device)
        final_matrix.append(get_evaluation_score_dssim(reconstructed_images, training_images, ds_mean=0))
    final_matrix = torch.cat(final_matrix, dim=1)
    return final_matrix


def get_total_successful_reconstructions(path_to_reconstructions_folder: Path, path_to_training_images_file: Path,
                                         threshold=0.4, device='cuda:0') -> Tensor:
    dssim_matrix = get_dssim_matrix_for_all_attacks(path_to_reconstructions_folder, path_to_training_images_file,
                                                    device)
    dssim_success_matrix = dssim_matrix < threshold
    number_of_vulnerable_training_images = (dssim_success_matrix.sum(dim=0) > 0).sum()
    return number_of_vulnerable_training_images.detach().item()


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
    if not isinstance(args, list):
        args = args[0]
    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    print(get_total_successful_reconstructions(Path(args.reconstruction_folder), Path(args.train_file)))
