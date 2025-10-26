from pathlib import Path

import torch


def get_dataloader(args):
    # for legacy:
    args.input_dim = 100
    args.num_classes = 2
    args.output_dim = 1
    args.dataset = 'sphere'
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    # Data already has mean 0
    args.data_reduce_mean = False

    if args.run_mode == 'reconstruct' or args.run_mode == 'train_reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    data_path = Path(args.datasets_dir) / 'sphere'
    if data_path.exists() and data_path.is_dir() and any(data_path.iterdir()):  # Check if dir exists and not empty
        train_data = torch.load(data_path / 'train_data.pt')
        X_train, y_train = train_data['data'], train_data['labels']
        test_data = torch.load(data_path / 'test_data.pt')
        X_test, y_test = test_data['data'], test_data['labels']
    else:
        # Step 1: Sample from a standard normal distribution
        total_amount = args.data_amount + args.data_test_amount
        X = torch.randn(total_amount, args.input_dim, device=args.device)

        # Step 2: Normalize to have unit norm (i.e., project to the unit sphere)
        X = X / X.norm(dim=1, keepdim=True)

        # Step 3: Compute XOR of sign bits
        # sign(x) > 0 => 1, sign(x) < 0 => 0
        signs = (X > 0).to(torch.int)
        # XOR across coordinates → parity (sum mod 2)
        y = signs.sum(dim=1) % 2
        y = y.to(torch.get_default_dtype())

        # Step 4: Split into train/test
        # Shuffle before splitting to ensure randomness
        indices = torch.randperm(total_amount, device=args.device)
        train_idx, test_idx = indices[:args.data_amount], indices[args.data_amount:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Create dir and save dataset
        data_path.mkdir(parents=True, exist_ok=True)
        torch.save({"data": X_train, "labels": y_train}, data_path / 'train_data.pt')
        torch.save({"data": X_test, "labels": y_test}, data_path / 'test_data.pt')

    return [(X_train, y_train)], [(X_test, y_test)], None
