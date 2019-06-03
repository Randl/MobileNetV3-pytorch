# https://github.com/eladhoffer/utils.pytorch/blob/ca6a47a7766c50930a607d8425216d39104b7664/optim.py

from copy import deepcopy

import torch
from torch.optim.lr_scheduler import CyclicLR

from cosine_with_warmup import CosineLR


def copy_params(param_target, param_src):
    with torch.no_grad():
        for p_src, p_target in zip(param_src, param_target):
            p_target.copy_(p_src)


def copy_params_grad(param_target, param_src):
    for p_src, p_target in zip(param_src, param_target):
        if p_target.grad is None:
            p_target.backward(p_src.grad.to(dtype=p_target.dtype))
        else:
            p_target.grad.detach().copy_(p_src.grad)


class ModuleFloatShadow(torch.nn.Module):
    def __init__(self, module):
        super(ModuleFloatShadow, self).__init__()
        self.original_module = module
        self.float_module = deepcopy(module)
        self.float_module.to(dtype=torch.float32)

    def parameters(self, *kargs, **kwargs):
        return self.float_module.parameters(*kargs, **kwargs)

    def named_parameters(self, *kargs, **kwargs):
        return self.float_module.named_parameters(*kargs, **kwargs)

    def modules(self, *kargs, **kwargs):
        return self.float_module.modules(*kargs, **kwargs)

    def named_modules(self, *kargs, **kwargs):
        return self.float_module.named_modules(*kargs, **kwargs)

    def original_parameters(self, *kargs, **kwargs):
        return self.original_module.parameters(*kargs, **kwargs)

    def original_named_parameters(self, *kargs, **kwargs):
        return self.original_module.named_parameters(*kargs, **kwargs)

    def original_modules(self, *kargs, **kwargs):
        return self.original_module.modules(*kargs, **kwargs)

    def original_named_modules(self, *kargs, **kwargs):
        return self.original_module.named_modules(*kargs, **kwargs)


class OptimizerWrapper(object):
    def __init__(self, model, optimizer_class, optimizer_params, scheduler_class, scheduler_params,
                 optimizer_state_dict=None, use_shadow_weights=False):
        if use_shadow_weights:
            model = ModuleFloatShadow(model)
            self._original_parameters = list(model.original_parameters())

        self.parameters = list([p for p in model.parameters() if p.requires_grad])
        self.optimizer = optimizer_class(self.parameters, **optimizer_params)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
        self.use_shadow_weights = use_shadow_weights

    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`.
        """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        optimizer_state_dict = state_dict['state']
        self.optimizer.__setstate__(optimizer_state_dict)

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        self.optimizer.zero_grad()
        if self.use_shadow_weights:
            for p in self._original_parameters:
                if p.grad is not None:
                    p.grad.detach().zero_()

    def optimizer_step(self, closure=None):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        if self.use_shadow_weights:
            copy_params_grad(self.parameters, self._original_parameters)
        self.optimizer.step(closure)
        if self.use_shadow_weights:
            copy_params(self._original_parameters, self.parameters)

    def scheduler_step(self, epoch=None):
        """Performs a single lr update step.
        """
        self.scheduler.step()

    def batch_step(self, closure=None):
        if isinstance(self.scheduler, CyclicLR) or isinstance(self.scheduler, CosineLR):
            self.scheduler_step()
        self.optimizer_step(closure)

    def epoch_step(self):
        if not isinstance(self.scheduler, CyclicLR) and not isinstance(self.scheduler, CosineLR):
            self.scheduler_step()
