import torch

from adversarialTraining import get_adv_auto_attack
from utils.utils import normalize_images


def get_margin(args, model, data_loader, compute_for_adv=False):
    model.eval()
    margin = float('inf')
    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        y = 2 * y - 1
        if compute_for_adv and args.train_robust and args.train_robust_radius > 0:
            x = get_adv_auto_attack(args, model, x, y)
        if args.data_reduce_mean:
            x = normalize_images(x, mean=args.mean, std=args.std)
        candidate_for_margin = torch.min(y * model(x).squeeze()).squeeze().cpu().item()
        if candidate_for_margin < margin:
            margin = candidate_for_margin
    return margin


def get_distances_from_margin(args, margin, model, data_loader, compute_for_adv=False):
    distances = []
    model.eval()
    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        y = 2 * y - 1
        if compute_for_adv and args.train_robust and args.train_robust_radius > 0:
            x = get_adv_auto_attack(args, model, x, y)
        if args.data_reduce_mean:
            x = normalize_images(x, mean=args.mean, std=args.std)
        distances.append((y * model(x).squeeze()).squeeze().detach().cpu() - margin)
    return torch.cat(distances, dim=0)

