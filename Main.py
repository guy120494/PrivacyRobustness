import os
import random
import sys

import threadpoolctl
import torch
import numpy as np
import datetime
import wandb

import common_utils
from adversarialTraining import get_adv_examples
from common_utils.common import AverageValueMeter, load_weights, now, save_weights
from CreateData import setup_problem
from CreateModel import create_model
from extraction import calc_extraction_loss, evaluate_extraction, get_trainable_params
from GetParams import get_args
from utils import normalize_images, unnormalize_images, replace_relu_with_modified_relu, get_margin, \
    get_distances_from_margin

thread_limit = threadpoolctl.threadpool_limits(limits=8)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


###############################################################################
#                               Train                                         #
###############################################################################
def get_loss_ce(args, model, x, y):
    p = model(x)
    p = p.view(-1)
    loss = torch.nn.BCEWithLogitsLoss()(p, y)
    return loss, p


def get_total_err(args, p, y):
    # BCEWithLogitsLoss needs 0,1
    err = (p.sign().view(-1).add(1).div(2) != y).float().mean().item()
    return err


# def epoch_ce_sgd(args, dataloader, model, epoch, device, batch_size, opt=None):
#     total_loss, total_err = AverageValueMeter(), AverageValueMeter()
#     model.train()
#     for i, (x, y) in enumerate(dataloader):
#         idx = torch.randperm(len(x))
#         x, y = x[idx], y[idx]
#         x, y = x.to(device), y.to(device)
#         for batch_idx in range(len(x) // batch_size + 1):
#             batch_x, batch_y = x[batch_idx * batch_size: (batch_idx + 1) * batch_size], y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
#             if len(batch_x) == 0:
#                 continue
#             loss, p = get_loss_ce(args, model, x, y)
#
#             if opt:
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()
#
#             err = get_total_err(args, p, y)
#             total_err.update(err)
#
#             total_loss.update(loss.item())
#     return total_err.avg, total_loss.avg, p.data


def epoch_ce(args, dataloader, model, device, opt=None, is_train=True):
    total_loss, total_err = AverageValueMeter(), AverageValueMeter()
    model.train()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        if is_train and args.train_robust:
            if args.train_add_adv_examples:
                x_adv = get_adv_examples(args, model, x, y)
                x = torch.cat([x, x_adv], dim=0)
                y = torch.cat([y, y], dim=0)
            else:
                x = get_adv_examples(args, model, x, y)
        if args.data_reduce_mean:
            x = normalize_images(x, mean=args.mean, std=args.std)
        loss, p = get_loss_ce(args, model, x, y)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        err = get_total_err(args, p, y)
        total_err.update(err)

        total_loss.update(loss.item())

    if args.data_reduce_mean:
        x = unnormalize_images(x, mean=args.mean, std=args.std)
    return total_err.avg, total_loss.avg, p.data, x


def train(args, train_loader, test_loader, val_loader, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.train_lr, weight_decay=args.train_weight_decay)
    print('Model:')
    print(model)
    x, _ = next(iter(train_loader))
    if args.data_reduce_mean:
        args.mean = x.mean(dim=[0, -2, -1]).detach()
        # args.std = x.std(dim=[0, -2, -1]).detach()
        args.std = torch.ones_like(x.mean(dim=[0, -2, -1])).detach()  # Should work better with KKT
    for epoch in range(args.train_epochs + 1):
        # if args.train_SGD:
        #     train_error, train_loss, output = epoch_ce_sgd(args, train_loader, model, epoch, args.device, args.train_SGD_batch_size, optimizer)
        # else:
        train_error, train_loss, output, x = epoch_ce(args, train_loader, model, args.device, optimizer)

        if epoch % args.train_evaluate_rate == 0:
            test_error, test_loss, _, _ = epoch_ce(args, test_loader, model, args.device, None, False)
            if val_loader is not None:
                validation_error, validation_loss, _, _ = epoch_ce(args, val_loader, model, args.device, None, False)
                print(now(),
                      f'Epoch {epoch}: train-loss = {train_loss:.8g} ; train-error = {train_error:.4g} ; test-loss = {test_loss:.8g} ; test-error = {test_error:.4g} ; validation-loss = {validation_loss:.8g} ; validation-error = {validation_error:.4g} ; p-std = {output.abs().std()}; p-val = {output.abs().mean()}')
            else:
                print(now(),
                      f'Epoch {epoch}: train-loss = {train_loss:.8g} ; train-error = {train_error:.4g} ; test-loss = {test_loss:.8g} ; test-error = {test_error:.4g} ; p-std = {output.abs().std()}; p-val = {output.abs().mean()}')

            if args.wandb_active:
                wandb.log(
                    {"epoch": epoch, "train loss": train_loss, 'train error': train_error, 'p-val': output.abs().mean(),
                     'p-std': output.abs().std()})
                if val_loader is not None:
                    wandb.log({'validation loss': validation_loss, 'validation error': validation_error})
                if len(x.shape) > 2:
                    wandb.log({'adversarial images': wandb.Image(
                        x[random.randint(0, args.data_amount - 1)].detach().cpu().numpy().transpose([1, 2, 0]),
                        caption=f'Adversarial')})
                wandb.log({'test loss': test_loss, 'test error': test_error})

        if np.isnan(train_loss):
            raise ValueError('Optimizer diverged')

        if train_loss < args.train_threshold:
            print(f'Reached train threshold {args.train_threshold} (train_loss={train_loss})')
            break

        if args.train_save_model_every > 0 and epoch % args.train_save_model_every == 0:
            save_weights(os.path.join(args.output_dir, 'weights'), model, ext_text=args.model_name, epoch=epoch)

    print(now(), 'ENDED TRAINING')
    return model


###############################################################################
#                               Extraction                                    #
###############################################################################

def data_extraction(args, dataset_loader, model):
    # we use dataset only for shapes and post-visualization (adding mean if it was reduced)
    x0, y0 = next(iter(dataset_loader))
    print('X:', x0.shape, x0.device)
    print('y:', y0.shape, y0.device)
    print('model device:', model.layers[0].weight.device)

    if args.data_reduce_mean and "mean" not in args:
        args.mean = x0.mean(dim=[0, -2, -1]).detach()
        # args.std = x.std(dim=[0, -2, -1]).detach()
        args.std = torch.ones_like(x0.mean(dim=[0, -2, -1])).detach()  # Should work better with KKT

    if args.data_reduce_mean:
        x0 = normalize_images(x0, mean=args.mean, std=args.std)

    # # send inputs to wandb/notebook
    # if args.wandb_active:
    #     send_input_data(args, model, x0, y0)

    # create labels (equal number of 1/-1)
    if args.extraction_random_init:
        y = torch.zeros(args.extraction_data_amount).type(torch.get_default_dtype()).to(args.device)
    else:
        y = torch.zeros(x0.shape[0]).type(torch.get_default_dtype()).to(args.device)
    y[:y.shape[0] // 2] = -1
    y[y.shape[0] // 2:] = 1
    y = y.long()

    # trainable parameters
    l, opt_l, opt_x, x = get_trainable_params(args, x0)

    print('y type,shape:', y.type(), y.shape)
    print('l type,shape:', l.type(), l.shape)

    torch.save(y, os.path.join(args.output_dir, "y.pth"))
    if args.wandb_active:
        wandb.save(os.path.join(wandb.run.dir, "y.pth"), base_path=args.wandb_base_path)

    # extraction phase
    for epoch in range(args.extraction_epochs):
        values = model(x).squeeze()
        loss, kkt_loss, cos_sim, loss_verify = calc_extraction_loss(args, l, model, values, x, y)
        if np.isnan(kkt_loss.item()):
            raise ValueError('Optimizer diverged during extraction')
        opt_x.zero_grad()
        opt_l.zero_grad()
        loss.backward()
        opt_x.step()
        opt_l.step()

        if epoch % args.extraction_evaluate_rate == 0:
            extraction_score = evaluate_extraction(args, epoch, kkt_loss, cos_sim, loss_verify, x, x0)
            if epoch >= args.extraction_stop_threshold and extraction_score > 3300 and args.problem != "sphere":
                print('Extraction Score is too low. Epoch:', epoch, 'Score:', extraction_score)
                break

        # send extraction output to wandb
        # if (args.extract_save_results_every > 0 and epoch % args.extract_save_results_every == 0) \
        #         or (args.extract_save_results and epoch % args.extraction_evaluate_rate == 0):
        #     torch.save(x, os.path.join(args.output_dir, 'x', f'{epoch}_x.pth'))
        #     torch.save(l, os.path.join(args.output_dir, 'l', f'{epoch}_l.pth'))
        #     if args.wandb_active:
        #         wandb.save(os.path.join(args.output_dir, 'x', f'{epoch}_x.pth'), base_path=args.wandb_base_path)
        #         wandb.save(os.path.join(args.output_dir, 'l', f'{epoch}_l.pth'), base_path=args.wandb_base_path)

    if args.extract_save_results:
        torch.save(x, os.path.join(args.output_dir, 'x', f'x_final.pth'))
        torch.save(l, os.path.join(args.output_dir, 'l', f'l_final.pth'))


###############################################################################
#                               MAIN                                          #
###############################################################################
def create_dirs_save_files(args):
    if args.train_save_model or args.extract_save_results:
        # create dirs
        os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'x'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'l'), exist_ok=True)

    if args.save_args_files:
        # save args
        common_utils.common.dump_obj_with_dict(args, f"{args.output_dir}/args.txt")
        # save command line
        with open(f"{args.output_dir}/sys.args.txt", 'w') as f:
            f.write(" ".join(sys.argv))


def setup_args(args):
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    from settings import datasets_dir, models_dir, results_base_dir
    args.results_base_dir = results_base_dir
    args.datasets_dir = datasets_dir
    if args.pretrained_model_path:
        args.pretrained_model_path = os.path.join(models_dir, args.pretrained_model_path)
    args.model_name = f'{args.problem}_d{args.data_per_class_train}'
    if args.proj_name:
        args.model_name += f'_{args.proj_name}'
    if args.pretrained_model_path:
        args.model_name = os.path.basename(args.pretrained_model_path)
        args.model_name = os.path.splitext(args.model_name)[0]
        if args.proj_name:
            args.model_name += f'_{args.proj_name}'

    torch.manual_seed(args.seed)

    if args.wandb_active:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity)
        wandb.config.update(args)

    if args.wandb_active and False:
        args.output_dir = wandb.run.dir
    else:
        import dateutil.tz
        timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
        run_name = f'{timestamp}_{np.random.randint(1e5, 1e6)}_{args.model_name}'
        args.output_dir = os.path.join(args.results_base_dir, args.model_name, run_name)
    print('OUTPUT_DIR:', args.output_dir)

    args.wandb_base_path = './'

    return args


def get_robustness_error_and_accuracy(args, model, train_loader):
    total_err = AverageValueMeter()
    total_acc = AverageValueMeter()
    model.eval()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(args.device), y.to(args.device)
        x = get_adv_examples(args, model, x, y, radius=0.5)
        if args.data_reduce_mean:
            x = normalize_images(x, mean=args.mean, std=args.std)
        loss, p = get_loss_ce(args, model, x, y)
        err = get_total_err(args, p, y)
        total_err.update(err)
        total_acc.update((p.sign().view(-1).add(1).div(2) == y).float().mean().item())
    return total_err.avg, total_acc.avg


def main_train(args, train_loader, test_loader, val_loader):
    print('TRAINING A MODEL')
    model = create_model(args, extraction=False)
    if args.wandb_active:
        wandb.watch(model)

    trained_model = train(args, train_loader, test_loader, val_loader, model)
    train_robust_error, train_robust_accuracy = get_robustness_error_and_accuracy(args, trained_model, train_loader)
    if args.wandb_active:
        wandb.log({"train robustness error": train_robust_error, "train robustness accuracy": train_robust_accuracy})
    else:
        print(f"train robustness error: {train_robust_error}")
        print(f"train robustness accuracy: {train_robust_accuracy}")

    test_robust_error, test_robust_accuracy = get_robustness_error_and_accuracy(args, trained_model, test_loader)
    if args.wandb_active:
        wandb.log({"test robustness error": test_robust_error, "test robustness accuracy": test_robust_accuracy})
    else:
        print(f"test robustness error: {test_robust_error}")
        print(f"test robustness accuracy: {test_robust_accuracy}")

    if args.wandb_active:
        margin = get_margin(args, trained_model, train_loader)
        distances = get_distances_from_margin(args, margin, trained_model, train_loader)
        wandb.log({"margin": margin, "min distance from margin": torch.min(distances).cpu().squeeze().item(),
                   "average distance from margin": torch.mean(distances).cpu().squeeze().item(),
                   "max distance from margin": torch.max(distances).cpu().squeeze().item(),
                   "number of points with minimum distance": (
                           distances == distances.min()).sum().cpu().squeeze().item()})

    if args.train_save_model:
        save_weights(args.output_dir, trained_model, ext_text=args.model_name)
    return model


def main_reconstruct(args, train_loader):
    print('USING PRETRAINED MODEL AT:', args.pretrained_model_path)
    extraction_model = create_model(args, extraction=True)
    extraction_model.eval()
    extraction_model = load_weights(extraction_model, args.pretrained_model_path, device=args.device)
    print('EXTRACTION MODEL:')
    print(extraction_model)

    data_extraction(args, train_loader, extraction_model)


def validate_settings_exists():
    if os.path.isfile("settings.py"):
        return
    raise FileNotFoundError("You should create a 'settings.py' file with the contents of 'settings.deafult.py', " +
                            "adjusted according to your system")


def train_and_extract(args, train_loader, test_loader, val_loader):
    print('TRAIN AND EXTRACT')
    trained_model = main_train(args, train_loader, test_loader, val_loader)
    print('START EXTRACTING')
    trained_model = replace_relu_with_modified_relu(args, trained_model)
    trained_model.eval()
    data_extraction(args, train_loader, trained_model)


def main():
    print(now(), 'STARTING!')
    validate_settings_exists()
    args = get_args(sys.argv[1:])
    args = setup_args(args)
    create_dirs_save_files(args)
    print('ARGS:')
    print(args)
    print('*' * 100)

    if args.cuda:
        print(f'os.environ["CUDA_VISIBLE_DEVICES"]={os.environ["CUDA_VISIBLE_DEVICES"]}')
    if args.precision == 'double':
        torch.set_default_dtype(torch.float64)

    print('DEVICE:', args.device)
    print('DEFAULT DTYPE:', torch.get_default_dtype())

    train_loader, test_loader, val_loader = setup_problem(args)

    # train
    if args.run_mode == 'train':
        main_train(args, train_loader, test_loader, val_loader)
    # reconstruct
    elif args.run_mode == 'reconstruct':
        main_reconstruct(args, train_loader)
    elif args.run_mode == 'train_reconstruct':
        train_and_extract(args, train_loader, test_loader, val_loader)
    else:
        raise ValueError(f'no such args.run_mode={args.run_mode}')


if __name__ == '__main__':
    main()
