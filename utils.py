import torch

from CreateModel import get_activation
from adversarialTraining import get_adv_auto_attack


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


def replace_relu_with_modified_relu(args, model):
    """
    Replace all instances of nn.ReLU in a model with ModifiedReLU (threshold=300).

    Args:
        model (nn.Module): The PyTorch model to modify

    Returns:
        nn.Module: The modified model
    """
    for name, module in model.named_children():
        # If the module is a ReLU
        if isinstance(module, torch.nn.ReLU):
            # Create and set the ModifiedReLU
            setattr(model, name, get_activation(args.extraction_model_activation, args.extraction_model_relu_alpha))
        # If the module has children, recursively process them
        elif len(list(module.children())) > 0:
            replace_relu_with_modified_relu(args, module)

    return model


def normalize_images(images, mean=None, std=None):
    """
    Normalize a batch of images using the provided mean and standard deviation.

    Args:
        images (torch.Tensor): Tensor of images with shape [B, C, H, W] in range [0, 1]
        mean (list or float): Channel-wise mean values for normalization.
                             If None, uses ImageNet mean for 3-channel or 0.5 for single-channel.
        std (list or float): Channel-wise standard deviation values.
                            If None, uses ImageNet std for 3-channel or 0.5 for single-channel.

    Returns:
        torch.Tensor: Normalized images with shape [B, C, H, W]
    """
    # Ensure inputs are tensors with proper dimensions
    if not isinstance(images, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    if len(images.shape) != 4:
        raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {images.shape}")

    # Get number of channels
    num_channels = images.shape[1]

    # Set default normalization parameters based on number of channels
    if mean is None:
        mean = [0.485, 0.456, 0.406] if num_channels == 3 else [0.5] * num_channels
    if std is None:
        std = [0.229, 0.224, 0.225] if num_channels == 3 else [0.5] * num_channels

    # If mean/std are provided as single values, expand to match channels
    if isinstance(mean, (int, float)):
        mean = [mean] * num_channels
    if isinstance(std, (int, float)):
        std = [std] * num_channels

    # Convert lists to tensors if needed
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=images.dtype, device=images.device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=images.dtype, device=images.device)

    # Ensure mean/std have correct number of values
    if len(mean) != num_channels:
        raise ValueError(f"Number of channels in images ({num_channels}) does not match length of mean ({len(mean)})")
    if len(std) != num_channels:
        raise ValueError(f"Number of channels in images ({num_channels}) does not match length of std ({len(std)})")

    # Reshape mean and std for proper broadcasting
    mean = mean.view(1, num_channels, 1, 1)
    std = std.view(1, num_channels, 1, 1)

    # Normalize
    normalized_images = (images - mean) / std

    return normalized_images


def unnormalize_images(normalized_images, mean=None, std=None):
    """
    Reverse the normalization process to recover the original images.

    Args:
        normalized_images (torch.Tensor): Tensor of normalized images with shape [B, C, H, W]
        mean (list or float): Channel-wise mean values used in normalization.
                             If None, uses ImageNet mean for 3-channel or 0.5 for single-channel.
        std (list or float): Channel-wise standard deviation values used in normalization.
                            If None, uses ImageNet std for 3-channel or 0.5 for single-channel.

    Returns:
        torch.Tensor: Unnormalized images with shape [B, C, H, W] in range [0, 1]
    """
    # Ensure inputs are tensors with proper dimensions
    if not isinstance(normalized_images, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    if len(normalized_images.shape) != 4:
        raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {normalized_images.shape}")

    # Get number of channels
    num_channels = normalized_images.shape[1]

    # Set default normalization parameters based on number of channels
    if mean is None:
        mean = [0.485, 0.456, 0.406] if num_channels == 3 else [0.5] * num_channels
    if std is None:
        std = [0.229, 0.224, 0.225] if num_channels == 3 else [0.5] * num_channels

    # If mean/std are provided as single values, expand to match channels
    if isinstance(mean, (int, float)):
        mean = [mean] * num_channels
    if isinstance(std, (int, float)):
        std = [std] * num_channels

    # Convert lists to tensors if needed
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=normalized_images.dtype, device=normalized_images.device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=normalized_images.dtype, device=normalized_images.device)

    # Ensure mean/std have correct number of values
    if len(mean) != num_channels:
        raise ValueError(f"Number of channels in images ({num_channels}) does not match length of mean ({len(mean)})")
    if len(std) != num_channels:
        raise ValueError(f"Number of channels in images ({num_channels}) does not match length of std ({len(std)})")

    # Reshape mean and std for proper broadcasting
    mean = mean.view(1, num_channels, 1, 1)
    std = std.view(1, num_channels, 1, 1)

    # Unnormalize
    original_images = normalized_images * std + mean

    # Clamp values to [0, 1] to ensure valid image range
    original_images = torch.clamp(original_images, 0.0, 1.0)

    return original_images


# Example usage
if __name__ == "__main__":
    # Test with 3-channel images (RGB)
    batch_size, height, width = 3, 24, 24
    rgb_images = torch.rand(batch_size, 3, height, width)

    # Test with 1-channel images (grayscale)
    grayscale_images = torch.rand(batch_size, 1, height, width)

    # Normalize and unnormalize RGB images using ImageNet stats
    rgb_normalized = normalize_images(rgb_images)
    rgb_reconstructed = unnormalize_images(rgb_normalized)

    # Normalize and unnormalize grayscale images using default values
    gray_normalized = normalize_images(grayscale_images)
    gray_reconstructed = unnormalize_images(gray_normalized)

    # Check if reconstructed images are close to original
    rgb_diff = torch.abs(rgb_images - rgb_reconstructed).max().item()
    gray_diff = torch.abs(grayscale_images - gray_reconstructed).max().item()

    print(f"RGB images - max difference: {rgb_diff}")
    print(f"Grayscale images - max difference: {gray_diff}")

    # Test with custom mean and std
    custom_mean = 0.3
    custom_std = 0.4
    custom_normalized = normalize_images(grayscale_images, mean=custom_mean, std=custom_std)
    custom_reconstructed = unnormalize_images(custom_normalized, mean=custom_mean, std=custom_std)
    custom_diff = torch.abs(grayscale_images - custom_reconstructed).max().item()
    print(f"Custom normalization - max difference: {custom_diff}")
