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


def get_adv_examples(args, model, x, y):
    """
    Ultra-optimized L2 PGD Attack using advanced PyTorch features

    Additional optimizations:
    1. torch.linalg.vector_norm for fastest norm computation
    2. Fused operations where possible
    3. Memory layout optimization
    4. Reduced conditional branches
    """
    batch_size = x.size(0)
    device = x.device

    # Use contiguous memory layout for better performance
    x = x.contiguous()

    # INITIALIZATION with torch.linalg.vector_norm (fastest)
    delta = torch.randn_like(x)

    # Reshape for vectorized operations
    delta_view = delta.view(batch_size, -1)
    delta_norm = torch.linalg.vector_norm(delta_view, dim=1, keepdim=True)
    delta_view /= delta_norm + 1e-12

    # Random radius scaling
    random_radius = torch.rand(batch_size, 1, device=device) * args.train_robust_radius
    delta_view *= random_radius

    # Initialize adversarial example
    x_adv = torch.clamp(x + delta, 0, 1)

    # Precompute constants
    alpha_tensor = torch.tensor(0.01, device=device)
    epsilon_tensor = torch.tensor(args.train_robust_radius, device=device)

    # Main PGD loop
    for _ in range(args.train_robust_epochs):
        x_adv.requires_grad_(True)

        # Forward pass with normalized input
        outputs = model(normalize_images(x_adv, args.mean, args.std)) if args.data_reduce_mean else model(x_adv)

        # Loss computation (avoid branching in loop)
        loss = F.cross_entropy(outputs.view(-1), y)

        # Backward pass
        loss.backward()

        # Gradient processing
        grad = x_adv.grad.data
        grad_view = grad.view(batch_size, -1)
        grad_norm = torch.linalg.vector_norm(grad_view, dim=1, keepdim=True)
        grad_view /= grad_norm + 1e-12

        # Clear gradients to prevent memory accumulation
        x_adv.grad.zero_()

        # Gradient step
        x_adv = x_adv + alpha_tensor * grad

        # L2 projection (vectorized)
        delta = x_adv - x
        delta_view = delta.view(batch_size, -1)
        delta_norm = torch.linalg.vector_norm(delta_view, dim=1, keepdim=True)

        # Project to L2 ball
        scale = torch.minimum(epsilon_tensor / (delta_norm + 1e-12),
                              torch.ones_like(delta_norm))
        delta_view *= scale

        # Update and clamp
        x_adv = torch.clamp(x + delta, 0, 1).detach()

    return x_adv
