import os
import shutil

import matplotlib
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm, trange

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from torch.optim.lr_scheduler import CyclicLR


def train(model, loader, mixup, epoch, optim, criterion, device, dtype, batch_size, log_interval, child):
    model.train()
    correct1, correct5 = 0, 0

    enum_load = enumerate(loader) if child else enumerate(tqdm(loader))
    for batch_idx, (data, t) in enum_load:
        data, t = data.to(device=device, dtype=dtype), t.to(device=device)
        data, target = mixup(data, t)

        optim.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optim.batch_step()

        corr = correct(output, t, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]
        if batch_idx % log_interval == 0 and not child:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'Top-1 accuracy: {:.2f}%({:.2f}%). '
                'Top-5 accuracy: {:.2f}%({:.2f}%).'.format(epoch, batch_idx, len(loader),
                                                           100. * batch_idx / len(loader), loss.item(),
                                                           100. * corr[0] / batch_size,
                                                           100. * correct1 / (batch_size * (batch_idx + 1)),
                                                           100. * corr[1] / batch_size,
                                                           100. * correct5 / (batch_size * (batch_idx + 1))))
    return loss.item(), correct1 / len(loader.sampler), correct5 / len(loader.sampler)


def test(model, loader, criterion, device, dtype, child):
    model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0

    enum_load = enumerate(loader) if child else enumerate(tqdm(loader))

    with torch.no_grad():
        for batch_idx, (data, target) in enum_load:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            corr = correct(output, target, topk=(1, 5))
            correct1 += corr[0]
            correct5 += corr[1]

    test_loss /= len(loader)
    if not child:
        tqdm.write(
            '\nTest set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
            'Top5: {}/{} ({:.2f}%)'.format(test_loss, int(correct1), len(loader.sampler),
                                           100. * correct1 / len(loader.sampler), int(correct5),
                                           len(loader.sampler), 100. * correct5 / len(loader.sampler)))
    return test_loss, correct1 / len(loader.sampler), correct5 / len(loader.sampler)


def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res


def save_checkpoint(state, is_best, filepath='./', filename='checkpoint{}.pth.tar', local_rank=0, test=False):
    save_path = os.path.join(filepath, filename.format(local_rank))
    best_path = os.path.join(filepath, 'model_best{}.pth.tar'.format(local_rank))
    torch.save(state, save_path)
    if test:
        torch.load(save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)


def swa_clr(folder, device):
    checkpoints = []
    for file in os.listdir(folder):
        if not 'SW' in file:
            continue
        fname = os.path.join(folder, file)
        checkpoints.append(torch.load(fname, map_location=device))
    sd = checkpoints[0]['state_dict'].copy()
    for i, cp in enumerate(checkpoints):
        for w in cp['state_dict']:
            sd[w] = i / (i + 1.) * sd[w] + 1. / (i + 1.) * cp['state_dict'][w]
    return sd


def find_bounds_clr(model, loader, optimizer, criterion, device, dtype, min_lr=8e-6, max_lr=8e-5, step_size=2000,
                    mode='triangular', save_path='.'):
    model.train()
    correct1, correct5 = 0, 0
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=step_size, mode=mode)
    epoch_count = step_size // len(loader)  # Assuming step_size is multiple of batch per epoch
    accuracy = []
    for _ in trange(epoch_count):
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            if scheduler is not None:
                scheduler.step()
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            corr = correct(output, target)
            accuracy.append(corr[0] / data.shape[0])

    lrs = np.linspace(min_lr, max_lr, step_size)
    plt.plot(lrs, accuracy)
    plt.show()
    plt.savefig(os.path.join(save_path, 'find_bounds_clr.pdf'))
    np.save(os.path.join(save_path, 'acc.npy'), accuracy)
    return
