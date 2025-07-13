from collections import namedtuple
import torch
# import foolbox as fb
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


def get_adv_examples(args, model, x, y, norm_type="l2", grad_norm="l2"):
    """
        Fast PGD attack implementation optimized for adversarial training.
        Memory efficient with minimal overhead.
        """

    # Initialize adversarial examples
    with torch.no_grad():
        x_adv = x.clone()

        # Random start within eps-ball
        if norm_type == 'linf':
            noise = torch.empty_like(x_adv).uniform_(-args.train_robust_radius, args.train_robust_radius)
        elif norm_type == 'l2':
            noise = torch.randn_like(x_adv)
            # Normalize to eps-ball
            noise_norm = noise.view(noise.size(0), -1).norm(p=2, dim=1, keepdim=True)
            noise_norm = noise_norm.view(-1, 1, 1, 1)
            noise = noise / (noise_norm + 1e-10) * args.train_robust_radius * torch.rand_like(noise_norm)

        x_adv.add_(noise)
        x_adv.clamp_(0, 1)

    # PGD iterations
    for _ in range(args.train_robust_epochs):
        x_adv.requires_grad_(True)

        # Forward pass with normalized input
        outputs = model(normalize_images(x_adv, args.mean, args.std)) if args.data_reduce_mean else model(x_adv)

        # Loss computation (avoid branching in loop)
        loss = F.cross_entropy(outputs.view(-1), y)

        # Backward pass
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        # Normalize gradient
        with torch.no_grad():
            if grad_norm == 'sign':
                grad_normalized = grad.sign()
            elif grad_norm == 'l2':
                grad_norm_val = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True)
                grad_norm_val = grad_norm_val.view(-1, 1, 1, 1)
                grad_normalized = grad / (grad_norm_val + 1e-10)
            elif grad_norm == 'linf':
                grad_normalized = grad / (grad.abs().max() + 1e-10)
            else:
                raise ValueError(f"Unsupported grad_norm: {grad_norm}")

            # Update adversarial examples
            x_adv.add_(0.01 * grad_normalized)

            # Project back to eps-ball around original input
            delta = x_adv - x

            if norm_type == 'linf':
                delta.clamp_(-args.train_robust_radius, args.train_robust_radius)
            elif norm_type == 'l2':
                delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True)
                delta_norm = delta_norm.view(-1, 1, 1, 1)
                delta = delta / (delta_norm + 1e-10) * torch.clamp(delta_norm, max=args.train_robust_radius)

            x_adv = x + delta

            # Clip to valid range
            x_adv.clamp_(0, 1)

    return x_adv.detach()
