# temporary file
import math

from torch.optim.lr_scheduler import _LRScheduler


class CosineLR(_LRScheduler):
    """
    """

    def __init__(self, optimizer, max_epochs, warmup_epochs, iter_in_epoch, eta_min=0, last_epoch=-1):
        self.T_max = (max_epochs - warmup_epochs) * iter_in_epoch
        self.T_warmup = warmup_epochs * iter_in_epoch
        self.eta_min = eta_min
        self.warmup_step = eta_min
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_warmup == 0 or self.last_epoch > self.T_warmup:
            curr_T = self.last_epoch - self.T_warmup
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * curr_T / self.T_max)) / 2
                    for base_lr in self.base_lrs]
        else:
            return [(base_lr - self.eta_min) * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
