from collections import namedtuple
import torch
# import foolbox as fb
import torch.nn.functional as F
from autoattack.autopgd_base import APGDAttack
from robustness.attacker import Attacker
from torch import nn
from torch.nn import BCEWithLogitsLoss

from utils import normalize_images


def get_adv_auto_attack(args, model, x, y, radius=None):
    class BinaryToTwoClassLogits(nn.Module):
        def __init__(self, original_model, original_args):
            super().__init__()
            self.original_model = original_model
            self.original_args = original_args

        def forward(self, x):
            if self.original_args.data_reduce_mean:
                x = normalize_images(x, self.original_args.mean, self.original_args.std)
            logit_pos = self.original_model(x)

            # ensure shape (N, 1)
            if logit_pos.dim() == 1:
                logit_pos = logit_pos.unsqueeze(1)

            # logits for [class 0, class 1]
            logits = torch.cat([-logit_pos, logit_pos], dim=1)
            return logits

    eps = radius if radius is not None else args.train_robust_radius
    adversary = APGDAttack(BinaryToTwoClassLogits(model, args), norm='L2', n_iter=1, eps=eps, device=args.device)
    adversary.init_hyperparam(x)
    _, _, _, x_adv = adversary.attack_single_run(x, y)
    return x_adv


def get_adv_examples_madrylab(args, model, x, y):
    def get_loss_for_adv_examples(model, x, y):
        p = model(x)
        p = p.view(-1)
        loss = torch.nn.BCEWithLogitsLoss(reduction='none')(p, y)
        return loss, None

    MyDataset = namedtuple('MyDataset', 'mean std')
    adv_model = Attacker(model, MyDataset(mean=args.mean, std=args.std))
    adv_x = adv_model(x, y, should_normalize=args.data_reduce_mean, constraint="2", eps=args.train_robust_radius,
                      step_size=args.train_robust_radius, iterations=args.train_robust_epochs, do_tqdm=False,
                      custom_loss=get_loss_for_adv_examples)
    return adv_x


# def get_adv_examples_foolbox(args, model, x, y):
#     preprocessing = dict(mean=args.mean, std=args.std, axis=-3)
#     bounds = (0, 1)
#     if args.data_reduce_mean:
#         fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
#     else:
#         fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=None)
#
#     attack = fb.attacks.L2AdamPGD(steps=args.train_robust_epochs)
#     _, clipped, _ = attack(fmodel, x, y, epsilons=args.train_robust_radius)
#     return clipped.detach()


def get_adv_examples(args, model, x, y, norm_type="l2", grad_norm="l2", radius=None):
    """
        Fast PGD attack implementation optimized for adversarial training.
        Memory efficient with minimal overhead.
        """

    # Initialize adversarial examples
    if radius is None:
        radius = args.train_robust_radius
    with torch.no_grad():
        x_adv = x.clone()

        # Random start within eps-ball
        if norm_type == 'linf':
            noise = torch.empty_like(x_adv).uniform_(-radius, radius)
        elif norm_type == 'l2':
            noise = torch.randn_like(x_adv)
            # Normalize to eps-ball
            noise_norm = noise.view(noise.size(0), -1).norm(p=2, dim=1, keepdim=True)
            if len(noise_norm.shape) != len(noise.shape):
                noise_norm = noise_norm.view(-1, 1, 1, 1)
            noise = noise / (noise_norm + 1e-10) * radius * torch.rand_like(noise_norm)

        x_adv.add_(noise)
        if args.problem != "sphere":
            x_adv.clamp_(0, 1)

    # PGD iterations
    for _ in range(args.train_robust_epochs):
        x_adv.requires_grad_(True)

        # Forward pass with normalized input
        outputs = model(normalize_images(x_adv, args.mean, args.std)) if args.data_reduce_mean else model(x_adv)

        # Loss computation (avoid branching in loop)
        loss = BCEWithLogitsLoss()(outputs.view(-1), y)

        # Backward pass
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        # Normalize gradient
        with torch.no_grad():
            if grad_norm == 'sign':
                grad_normalized = grad.sign()
            elif grad_norm == 'l2':
                grad_norm_val = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True)
                if len(grad_norm_val.shape) != len(grad.shape):
                    grad_norm_val = grad_norm_val.view(-1, 1, 1, 1)
                grad_normalized = grad / (grad_norm_val + 1e-10)
            elif grad_norm == 'linf':
                grad_normalized = grad / (grad.abs().max() + 1e-10)
            else:
                raise ValueError(f"Unsupported grad_norm: {grad_norm}")

            # Update adversarial examples
            x_adv.add_(args.train_robust_radius * grad_normalized)

            # Project back to eps-ball around original input
            delta = x_adv - x

            if norm_type == 'linf':
                delta.clamp_(-radius, radius)
            elif norm_type == 'l2':
                delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True)
                if len(delta_norm.shape) != len(delta.shape):
                    delta_norm = delta_norm.view(-1, 1, 1, 1)
                delta = delta / (delta_norm + 1e-10) * torch.clamp(delta_norm, max=radius)

            x_adv = x + delta

            # Clip to valid range
            if args.problem != "sphere":
                x_adv.clamp_(0, 1)

    return x_adv.detach()
