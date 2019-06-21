import argparse
import csv
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CyclicLR
from tqdm import trange

import flops_benchmark
from MobileNetV3 import MobileNetV3
from cosine_with_warmup import CosineLR
from data import get_loaders
from mixup import MixupScheduled
from run import train, test, save_checkpoint, find_bounds_clr, swa_clr
from utils.cross_entropy import CrossEntropyLoss
from utils.logger import CsvLogger
from utils.optimizer_wrapper import OptimizerWrapper

# https://arxiv.org/abs/1905.02244
# input_size, scale, large/small
# TODO
claimed_acc_top1 = {
    'large': {256: {1.: 0.76}, 224: {0.35: 0.642, 0.5: 0.688, 0.75: 0.733, 1.: 0.752, 1.25: 0.766}, 192: {1.: 0.737},
              160: {1.: 0.717}, 128: {1.: 0.684}, 96: {1.: 0.633}},
    'small': {256: {1.: 0.685}, 224: {0.35: 0.498, 0.5: 0.58, 0.75: 0.654, 1.: 0.675, 1.25: 0.704}, 192: {1.: 0.654},
              160: {1.: 0.628}, 128: {1.: 0.573}, 96: {1.: 0.517}}}


# TODO add hubconf.py https://pytorch.org/blog/towards-reproducible-research-with-pytorch-hub/

def get_args():
    parser = argparse.ArgumentParser(description='MobileNetV3 training with PyTorch')
    parser.add_argument('--dataroot', required=True, metavar='PATH',
                        help='Path to ImageNet train and val folders, preprocessed as described in '
                             'https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('-j', '--workers', default=7, type=int, metavar='N',
                        help='Number of data loading workers (default: 6)')
    parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')

    # distributed
    parser.add_argument('--world-size', default=-1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int, help='rank of distributed processes')
    parser.add_argument('--dist-init', default='env://', type=str, help='init used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')

    # Optimization options
    parser.add_argument('--sched', dest='sched', type=str, default='multistep')
    parser.add_argument('--epochs', type=int, default=755, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.02, help='The learning rate for batch of 128 '
                                                                                 '(scaled for bigger/smaller batches).')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=2.5e-7, help='Weight decay for batch of 128 '
                                                                        '(scaled for bigger/smaller batches).')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[200, 300],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--step', type=int, default=40, help='Decrease learning rate each time.')
    parser.add_argument('--warmup', default=0, type=int, metavar='N', help='Warmup length')
    parser.add_argument('--mixup', type=float, default=0.2, help='Mixup gamma value.')
    parser.add_argument('--mixup-warmup', type=int, default=455, help='Mixup time to be turned of in the beginning.')
    parser.add_argument('--smooth-eps', type=float, default=0.1, help='Label smoothing epsilon value.')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes.')

    # CLR
    parser.add_argument('--min-lr', type=float, default=2.5e-6, help='Minimal LR for CLR.')
    parser.add_argument('--max-lr', type=float, default=0.225, help='Maximal LR for CLR for batch of 128 '
                                                                    '(scaled for bigger/smaller batches).')
    parser.add_argument('--epochs-per-step', type=int, default=125,
                        help='Number of epochs per step in CLR, recommended to be between 2 and 10.')
    parser.add_argument('--mode', default='triangular2', help='CLR mode. One of {triangular, triangular2, exp_range}')
    parser.add_argument('--find-clr', dest='find_clr', action='store_true',
                        help='Run search for optimal LR in range (min_lr, max_lr)')

    # Checkpoints
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
    parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--swa', default='', type=str, metavar='PATH',
                        help='path to SWA folder (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='Number of batches between log messages')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: random)')
    parser.add_argument('--determenistic', dest='deter', action='store_true', help='use determenistic environment')
    parser.add_argument('--grad-debug', dest='grad_debug', action='store_true', help='use anomaly detection')

    # Architecture
    parser.add_argument('--scaling', type=float, default=1, metavar='SC', help='Scaling of MobileNetV3 (default x1).')
    parser.add_argument('--dp', type=float, default=0.1, metavar='DP', help='Dropping probability of DropBlock')
    parser.add_argument('--input-size', type=int, default=224, metavar='I', help='Input size of MobileNetV3.')
    parser.add_argument('--small', dest='small', action='store_true', help='use small modification')
    parser.add_argument('--sync-bn', dest='sync_bn', action='store_true', help='use synchronized BN')

    args = parser.parse_args()

    args.distributed = args.local_rank >= 0 or args.world_size > 1
    args.child = args.distributed and args.local_rank > 0
    if not args.distributed:
        args.local_rank = 0
        args.world_size = 1
    if args.local_rank >= args.world_size:
        raise ValueError('World size inconsistent with local rank!')
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    args.save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(args.save_path) and not args.child:
        os.makedirs(args.save_path)

    if args.device == 'cuda' and torch.cuda.is_available():
        cudnn.enabled = True
        if args.deter:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.benchmark = True
        if args.grad_debug:
            torch.autograd.set_detect_anomaly(True)
        args.gpus = [args.local_rank]
        args.device = 'cuda:' + str(args.gpus[0])
        torch.cuda.set_device(args.gpus[0])
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpus = []
        args.device = 'cpu'

    if args.type == 'float64':
        args.dtype = torch.float64
    elif args.type == 'float32':
        args.dtype = torch.float32
    elif args.type == 'float16':
        args.dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8

    # Adjust lr for batch size
    args.learning_rate *= args.batch_size / 128. * args.world_size
    args.max_lr *= args.batch_size / 128. * args.world_size
    args.decay *= args.batch_size / 128. * args.world_size

    if not args.child:
        print("Random Seed: ", args.seed)
        print(args)
    return args


def is_bn(module):
    return isinstance(module, torch.nn.BatchNorm1d) or \
           isinstance(module, torch.nn.BatchNorm2d) or \
           isinstance(module, torch.nn.BatchNorm3d) or \
           isinstance(module, torch.nn.BatchNorm3d)


def main():
    import warnings

    # filter out corrupted images warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    args = get_args()
    device, dtype = args.device, args.dtype

    train_loader, val_loader = get_loaders(args.dataroot, args.batch_size, args.batch_size, args.input_size,
                                           args.workers, args.world_size, args.local_rank)
    args.num_batches = len(train_loader) * args.epochs
    args.start_step = len(train_loader) * args.start_epoch
    model = MobileNetV3(num_classes=args.num_classes, scale=args.scaling, in_channels=3, drop_prob=args.dp,
                        num_steps=args.num_batches, start_step=args.start_step, small=args.small)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    flops = flops_benchmark.count_flops(MobileNetV3, 2, device, dtype, args.input_size, 3, num_classes=args.num_classes,
                                        scale=args.scaling, drop_prob=args.dp, num_steps=args.num_batches,
                                        start_step=args.start_step, small=args.small)
    if not args.child:
        print(model)
        print('number of parameters: {}'.format(num_parameters))
        print('FLOPs: {}'.format(flops))
        print('Resuts saved to {}'.format(args.save_path))

    # define loss function (criterion) and optimizer
    criterion = CrossEntropyLoss()

    model, criterion = model.to(device=device, dtype=dtype), criterion.to(device=device, dtype=dtype)
    if args.dtype == torch.float16:
        for module in model.modules():  # FP batchnorm
            if is_bn(module):
                module.to(dtype=torch.float32)  # github.com/pytorch/pytorch/issues/20634

    if args.distributed:
        args.device_ids = [args.local_rank]
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_init, world_size=args.world_size,
                                rank=args.local_rank)
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        print('Node #{}'.format(args.local_rank))
    else:
        model = torch.nn.parallel.DataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.find_clr:
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.decay, nesterov=True)
        find_bounds_clr(model, train_loader, optimizer, criterion, device, dtype, min_lr=args.min_lr,
                        max_lr=args.max_lr, step_size=args.epochs_per_step * len(train_loader), mode=args.mode,
                        save_path=args.save_path)
        return

    best_test = 0

    # optionally resume from a checkpoint
    data = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            args.start_step = len(train_loader) * args.start_epoch
            optim, mixup = init_optimizer_and_mixup(args, train_loader, model, checkpoint['optimizer'])
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint{}.pth.tar'.format(args.local_rank))
            csv_path = os.path.join(args.resume, 'results{}.csv'.format(args.local_rank))
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = checkpoint['epoch']
            args.start_step = len(train_loader) * args.start_epoch
            optim, mixup = init_optimizer_and_mixup(args, train_loader, model, checkpoint['optimizer'])
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            data = []
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        optim, mixup = init_optimizer_and_mixup(args, train_loader, model)

    if args.evaluate:
        if args.swa:
            sd = swa_clr(args.swa, device)
            model.load_state_dict(sd)
        loss, top1, top5 = test(model, val_loader, criterion, device, dtype, args.child)  # TODO
        return

    csv_logger = CsvLogger(filepath=args.save_path, data=data, local_rank=args.local_rank)
    csv_logger.save_params(sys.argv, args)

    claimed_acc1 = None
    claimed_acc5 = None
    ntype = 'small' if args.small else 'large'
    if ntype in claimed_acc_top1:
        if args.input_size in claimed_acc_top1[ntype]:
            if args.scaling in claimed_acc_top1[ntype][args.input_size]:
                claimed_acc1 = claimed_acc_top1[ntype][args.input_size][args.scaling]
                if not args.child:
                    csv_logger.write_text('Claimed accuracy is {:.2f}% top-1'.format(claimed_acc1 * 100.))
    train_network(args.start_epoch, args.epochs, optim, model, train_loader, val_loader, criterion, mixup,
                  device, dtype, args.batch_size, args.log_interval, csv_logger, args.save_path, claimed_acc1,
                  claimed_acc5, best_test, args.local_rank, args.child)


def train_network(start_epoch, epochs, optim, model, train_loader, val_loader, criterion, mixup, device, dtype,
                  batch_size, log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5, best_test, local_rank,
                  child):
    my_range = range if child else trange
    for epoch in my_range(start_epoch, epochs + 1):
        train_loss, train_accuracy1, train_accuracy5, = train(model, train_loader, mixup, epoch, optim, criterion,
                                                              device, dtype, batch_size, log_interval, child)
        test_loss, test_accuracy1, test_accuracy5 = test(model, val_loader, criterion, device, dtype, child)
        optim.epoch_step()

        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_accuracy1, 'val_error5': 1 - test_accuracy5,
                          'val_loss': test_loss, 'train_error1': 1 - train_accuracy1,
                          'train_error5': 1 - train_accuracy5, 'train_loss': train_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optim.state_dict()}, test_accuracy1 > best_test, filepath=save_path,
                        local_rank=local_rank)
        # TODO: save on the end of the cycle

        csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5)

        if test_accuracy1 > best_test:
            best_test = test_accuracy1

    csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))


def init_optimizer_and_mixup(args, train_loader, model, optim_state_dict=None):
    optimizer_class = torch.optim.SGD
    optimizer_params = {"lr": args.learning_rate, "momentum": args.momentum, "weight_decay": args.decay,
                        "nesterov": True}

    if args.sched == 'clr':
        scheduler_class = CyclicLR
        scheduler_params = {"base_lr": args.min_lr, "max_lr": args.max_lr,
                            "step_size_up": args.epochs_per_step * len(train_loader), "mode": args.mode,
                            "last_epoch": args.start_step - 1}
    elif args.sched == 'multistep':
        scheduler_class = MultiStepLR
        scheduler_params = {"milestones": args.schedule, "gamma": args.gamma, "last_epoch": args.start_epoch - 1}
    elif args.sched == 'cosine':
        scheduler_class = CosineLR
        scheduler_params = {"max_epochs": args.epochs, "warmup_epochs": args.warmup, "iter_in_epoch": len(train_loader),
                            "last_epoch": args.start_step - 1}
    elif args.sched == 'gamma':
        scheduler_class = StepLR
        scheduler_params = {"step_size": 30, "gamma": args.gamma, "last_epoch": args.start_epoch - 1}
    else:
        raise ValueError('Wrong scheduler!')
    optim = OptimizerWrapper(model, optimizer_class=optimizer_class, optimizer_params=optimizer_params,
                             optimizer_state_dict=optim_state_dict, scheduler_class=scheduler_class,
                             scheduler_params=scheduler_params, use_shadow_weights=args.dtype == torch.float16)
    mixup_start = len(train_loader) * args.mixup_warmup
    mixup_nr = len(train_loader) * (args.epochs - args.mixup_warmup)
    mixup = MixupScheduled(start_gamma=0, stop_gamma=args.mixup, wait_steps=mixup_start, nr_steps=mixup_nr,
                           start_step=args.start_step, num_classes=args.num_classes, smooth_eps=args.smooth_eps)
    return optim, mixup


if __name__ == '__main__':
    main()
