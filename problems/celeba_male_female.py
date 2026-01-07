from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def move_to_type_device(x, y, device):
    print('X:', x.shape)
    print('y:', y.shape)
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    x, y = x.to(device), y.to(device)
    return x, y


def create_labels(y0):
    return y0 % 2


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


def load_celebA_male_female(args):
    """
    Loads the CelebA dataset, partitions by gender (Male=1, Female=0),
    and samples a balanced subset with 250 train and 250 test images per class.
    """
    root_dir = Path(args.datasets_dir)

    # === Step 1: Define transforms ===
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # === Step 2: Load CelebA ===
    dataset = datasets.CelebA(
        root=root_dir,
        split="all",
        target_type="attr",
        transform=transform,
        download=True,
    )

    attr_names = dataset.attr_names
    assert "Male" in attr_names, "CelebA must include the 'Male' attribute."
    attr_idx = attr_names.index("Male")

    # === Step 3: Separate by gender ===
    male_indices = [i for i, a in enumerate(dataset.attr) if a[attr_idx].item() == 1]
    female_indices = [i for i, a in enumerate(dataset.attr) if a[attr_idx].item() == 0]

    # === Step 4: Sample train/test subsets per class ===
    def sample_indices(indices, n_train, n_test):
        n_available = len(indices)
        total_needed = n_train + n_test
        if n_available < total_needed:
            print(f"⚠️ Only {n_available} samples available, reducing counts.")
            n_train = n_test = n_available // 2
        return indices[:n_train], indices[n_train:n_train + n_test]

    train_male, test_male = sample_indices(male_indices, args.data_per_class_train,
                                           args.data_test_amount // args.num_classes)
    train_female, test_female = sample_indices(female_indices, args.data_per_class_train,
                                               args.data_test_amount // args.num_classes)

    train_indices = train_male + train_female
    test_indices = test_male + test_female

    # === Step 5: Build subsets and DataLoaders ===
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=100, shuffle=True)
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
    args.dataset = 'celeba'

    if args.run_mode == 'reconstruct' or args.run_mode == 'train_reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 500

    data_loader = load_celebA_male_female(args)
    return data_loader
