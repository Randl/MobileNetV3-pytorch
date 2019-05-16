import numpy as np
import torch
from torch.distributions import Beta

from utils.cross_entropy import onehot


def mixup(x, y, num_classes, gamma, smooth_eps):
    if gamma == 0 and smooth_eps == 0:
        return x, y
    my = onehot(y, num_classes).to(x)
    true_class, false_class = 1. - smooth_eps * num_classes / (num_classes - 1), smooth_eps / (num_classes - 1)
    my = my * true_class + torch.ones_like(my) * false_class
    if gamma == 0:
        return x, my
    perm = torch.randperm(x.size(0))
    x2, y2 = x[perm], my[perm]

    m = Beta(torch.tensor([gamma]), torch.tensor([gamma]))
    lambdas = m.sample([x.size(0), 1, 1]).to(x)
    return x * (1 - lambdas) + x2 * lambdas, my * (1 - lambdas) + y2 * lambdas


class Mixup(torch.nn.Module):
    def __init__(self, num_classes=1000, gamma=0, smooth_eps=0):
        super(Mixup, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        return mixup(input, target, self.num_classes, self.gamma, self.smooth_eps)


class MixupScheduled(torch.nn.Module):
    def __init__(self, start_gamma, stop_gamma, wait_steps, nr_steps, start_step, num_classes=1000, smooth_eps=0):
        super(MixupScheduled, self).__init__()
        self.num_classes = num_classes
        self.gamma = start_gamma
        self.smooth_eps = smooth_eps
        self.wait_steps = wait_steps

        self.i = start_step
        self.gamma_values = np.linspace(start=start_gamma, stop=stop_gamma, num=nr_steps)  # todo: log?

    def forward(self, input, target):
        self.step()
        return mixup(input, target, self.num_classes, self.gamma, self.smooth_eps)

    def step(self):
        curr = self.i - self.wait_steps
        if curr > 0 and curr < len(self.gamma_values):
            self.gamma = self.gamma_values[curr]

        self.i += 1
