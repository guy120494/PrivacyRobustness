import itertools
import os
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive


def move_to_type_device(x, y, device):
    print('X:', x.shape)
    print('y:', y.shape)
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    x, y = x.to(device), y.to(device)
    return x, y


def create_labels(y0):
    first_label = y0[0]
    return y0 == first_label


def get_balanced_data(args, data_loader, data_amount):
    print('BALANCING DATASET...')
    # get balanced data
    data_amount_per_class = data_amount // 2

    labels_counter = {1: 0, 0: 0}
    x0, y0 = [], []
    got_enough = False
    for bx, by in data_loader:
        by = create_labels(by)
        for i in range(len(bx)):
            if labels_counter[int(by[i])] < data_amount_per_class:
                labels_counter[int(by[i])] += 1
                x0.append(bx[i])
                y0.append(by[i])
            if (labels_counter[0] >= data_amount_per_class) and (labels_counter[1] >= data_amount_per_class):
                got_enough = True
                break
        if got_enough:
            break
    x0, y0 = torch.stack(x0), torch.stack(y0)
    return x0, y0


def load_imagenet_data(args):
    """
    Loads 2 random classes (500 images each). If dataset not found,
    automatically downloads a small ImageNet-like dataset (Tiny ImageNet).

    Returns:
        dataloader (DataLoader): PyTorch DataLoader object.
        selected_class_names (list[str]): Names of selected classes.
    """
    root_dir = Path(args.datasets_dir) / "imagenet"

    # === Step 1: Download Tiny ImageNet if dataset does not exist ===
    if not root_dir.exists():
        print(f"Dataset not found in {root_dir}. Downloading Tiny ImageNet...")
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        root_dir.mkdir(parents=True, exist_ok=True)
        download_and_extract_archive(url, download_root=root_dir, extract_root=root_dir)
        extracted = root_dir / "tiny-imagenet-200"
        # Move contents of tiny-imagenet-200/ → root_dir/
        for item in extracted.iterdir():
            target = root_dir / item.name
            item.rename(target)
        extracted.rmdir()

    # === Step 2: Load dataset ===
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=str(root_dir / "train"), transform=transform)

    # === Step 3: Group indices by class ===
    class_to_indices = {cls_idx: [] for cls_idx in range(len(dataset.classes))}
    for i, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(i)

    # === Step 4: Choose 2 classes ===
    selected_classes = [51, 130]

    # === Step 5: Sample disjoint train/test subsets ===
    train_indices, test_indices = [], []
    for cls in selected_classes:
        cls_indices = class_to_indices[cls]
        n_available = len(cls_indices)
        total_needed = args.data_per_class_train + args.data_test_amount // args.num_classes

        if n_available < total_needed:
            print(f"⚠️ Class '{dataset.classes[cls]}' has only {n_available} samples; reducing counts.")
            total_needed = n_available
            split_point = total_needed // 2
            train_split = cls_indices[:split_point]
            test_split = cls_indices[split_point:]
        else:
            train_split = cls_indices[:args.data_per_class_train]
            test_split = cls_indices[
                         args.data_per_class_train:args.data_per_class_train + args.data_test_amount // args.num_classes]

        # zip_longest fills missing spots with None by default
        combined_iterator = itertools.zip_longest(train_indices, train_split, fillvalue=None)
        # chain.from_iterable flattens the iterator of tuples into a single sequence
        train_indices = [item for item in itertools.chain.from_iterable(combined_iterator) if item is not None]
        # zip_longest fills missing spots with None by default
        combined_iterator = itertools.zip_longest(test_indices, test_split, fillvalue=None)
        # chain.from_iterable flattens the iterator of tuples into a single sequence
        test_indices = [item for item in itertools.chain.from_iterable(combined_iterator) if item is not None]

    # === Step 6: Create subsets and loaders ===
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=100, shuffle=False)

    x0, y0 = get_balanced_data(args, train_loader, args.data_amount)
    x0_test, y0_test = get_balanced_data(args, test_loader, args.data_test_amount)

    x0, y0 = move_to_type_device(x0, y0, args.device)
    x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)

    return [(x0, y0)], [(x0_test, y0_test)], None


def get_dataloader(args):
    args.input_dim = 64 * 64 * 3
    args.num_classes = 2
    args.output_dim = 1
    args.dataset = 'imagenet'

    if args.run_mode == 'reconstruct' or args.run_mode == 'train_reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 500

    data_loader = load_imagenet_data(args)

    return data_loader
