from typing import Dict, Iterator, List
import torch
from source.optimizers import get_optimizer
from source.schedulers import CosineLRSchedulerWithWarmup
from runs.fine_tune.config import ArgsFineTuningDefaults


def get_optimizer_and_lr_sched(
    args: ArgsFineTuningDefaults,
    parameter_groups: (
        List[Dict[str, Iterator[torch.nn.Parameter] | float]]
        | List[Dict[str, List[torch.nn.Parameter] | float]]
    ),
    len_loader: int,
    d_lr_sched_warmup_epochs: int = 0,
):

    assert all(
        ("params" in pg.keys() and "weight_decay" in pg.keys())
        for pg in parameter_groups
    )
    assert args.lr is not None
    optimizer = get_optimizer(
        args.optimizer,
        parameter_groups,
        lr=args.lr,
        momentum=args.momentum,
        betas=(args.beta1, args.beta2),
    )

    lr_scheduler = CosineLRSchedulerWithWarmup(
        optimizer=optimizer,
        n_epochs=args.epochs,
        len_loader=len_loader,
        warmup_epochs=d_lr_sched_warmup_epochs,
        end_lrs=1e-6,
    )

    return optimizer, lr_scheduler
