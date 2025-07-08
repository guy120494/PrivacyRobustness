from collections import namedtuple
import torch
import foolbox as fb
import torch.nn.functional as F
from robustness.attacker import Attacker

from utils import normalize_images


def get_adv_examples_madrylab(args, model, x, y):
    def get_loss_for_adv_examples(model, x, y):
        p = model(x)
        p = p.view(-1)
        loss = torch.nn.BCEWithLogitsLoss(reduction='none')(p, y)
        return loss, None

    MyDataset = namedtuple('MyDataset', 'mean std')
    adv_model = Attacker(model, MyDataset(mean=args.mean, std=args.std))
    adv_x = adv_model(x, y, should_normalize=args.data_reduce_mean, constraint="2", eps=args.train_robust_radius,
                      step_size=args.train_robust_lr, iterations=args.train_robust_epochs, do_tqdm=False,
                      custom_loss=get_loss_for_adv_examples)
    return adv_x


def get_adv_examples_foolbox(args, model, x, y):
    preprocessing = dict(mean=args.mean, std=args.std, axis=-3)
    bounds = (0, 1)
    if args.data_reduce_mean:
        fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    else:
        fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=None)

    attack = fb.attacks.L2AdamPGD(steps=args.train_robust_epochs)
    _, clipped, _ = attack(fmodel, x, y, epsilons=args.train_robust_radius)
    return clipped.detach()


def get_adv_examples(args, model, x, y):
    """PGD attack under L2 norm"""
    mean = args.mean
    std = args.std

    # Generate random perturbation within L2 ball
    delta = torch.randn_like(x) * args.train_robust_radius
    delta = delta / torch.norm(delta.view(delta.shape[0], -1), dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    delta = delta * args.train_robust_radius * torch.rand(x.shape[0], 1, 1, 1, device=x.device)

    x_adv = x + delta
    x_adv = torch.clamp(x_adv, 0, 1)  # Assuming input is in [0,1]

    for _ in range(args.train_robust_epochs):
        x_adv.requires_grad_(True)

        # Forward pass
        outputs = model(normalize_images(x_adv, mean, std)) if args.data_reduce_mean else model(x_adv)

        # Calculate loss
        loss = F.cross_entropy(outputs.view(-1), y)

        # Backward pass
        loss.backward()

        # Get gradient
        grad = x_adv.grad.data

        # L2 gradient step
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1, keepdim=True)
        grad_norm = grad_norm.unsqueeze(-1).unsqueeze(-1)
        grad = grad / (grad_norm + 1e-12)  # Normalize gradient

        # Update adversarial example
        x_adv = x_adv + 0.01 * grad

        # Project back to L2 ball
        delta = x_adv - x
        delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=1, keepdim=True)
        delta_norm = delta_norm.unsqueeze(-1).unsqueeze(-1)
        delta = (delta / torch.clamp(delta_norm, min=args.train_robust_radius / 1000) *
                 torch.clamp(delta_norm, max=args.train_robust_radius))

        x_adv = x + delta
        x_adv = torch.clamp(x_adv, 0, 1)  # Ensure valid pixel range
        x_adv = x_adv.detach()

    return x_adv
