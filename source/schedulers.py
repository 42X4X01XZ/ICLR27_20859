from typing import List, Optional

import math

import torch


class CosineLRSchedulerWithWarmup:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        len_loader: int,
        warmup_epochs: int,
        end_lrs: Optional[List[float] | float] = None,
        start_epoch: int = 0,
        batch_size: Optional[int] = None,
    ):
        self.optimizer: torch.optim.Optimizer = optimizer
        self.base_lrs: List[float] = [pg["lr"] for pg in optimizer.param_groups]

        if end_lrs is not None and isinstance(end_lrs, float):
            end_lrs = [end_lrs] * len(self.base_lrs)

        self.end_lrs: List[float] = (
            [base_lr * 0.001 for base_lr in self.base_lrs]  # type: ignore
            if end_lrs is None
            else end_lrs
        )

        self.lrs: List[float] = self.base_lrs
        if batch_size is not None:
            self.lrs: List[float] = [
                (batch_size / 256) * base_lr for base_lr in self.base_lrs
            ]

        self.total_steps: int = n_epochs * len_loader
        self.warmup_steps: int = warmup_epochs * len_loader

        self.tr_step: int = 1 + start_epoch * len_loader

        # If we include a linear warmup, we need to set the initial lr to a small value
        if self.warmup_steps > 0:
            self.warmup_steps += 1
            for idx, param_group in enumerate(self.optimizer.param_groups):
                lrs = [lr / self.warmup_steps for lr in self.lrs]
                if "lr_scale" in param_group:
                    param_group["lr"] = lrs[idx] * param_group["lr_scale"]
                else:
                    param_group["lr"] = lrs[idx]

            self.tr_step += 1

    def step(self, return_lr: bool = False) -> Optional[List[float]]:
        if self.tr_step < self.warmup_steps:
            lrs = [lr * self.tr_step / self.warmup_steps for lr in self.lrs]
        else:
            q = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (self.tr_step - self.warmup_steps)
                    / (self.total_steps - self.warmup_steps)
                )
            )
            lrs = [
                lr * q + end_lr * (1 - q) for lr, end_lr in zip(self.lrs, self.end_lrs)
            ]
        for idx, param_group in enumerate(self.optimizer.param_groups):
            if "lr_scale" in param_group:
                param_group["lr"] = lrs[idx] * param_group["lr_scale"]
            else:
                param_group["lr"] = lrs[idx]

        self.tr_step += 1

        if return_lr:
            return lrs


class ConstantLR:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer: torch.optim.Optimizer = optimizer
        self.lrs: List[float] = [pg["lr"] for pg in self.optimizer.param_groups]

    def step(self, return_lr: bool = False) -> Optional[List[float]]:
        for idx, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lrs[idx]

        if return_lr:
            return self.lrs
