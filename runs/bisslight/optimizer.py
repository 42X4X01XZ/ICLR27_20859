from typing import List, Dict, Iterator, Tuple
import torch
from source.optimizers import get_optimizer, ADAM_BETAS_DEFAULTS
from source.schedulers import CosineLRSchedulerWithWarmup

from runs.bisslight.classification.mae_lora.config import (
    ArgsBiSSLightClassificationMAELora,
)
from runs.pretext.config import ArgsPretextDefaults


def get_lower_pretext_optimizer_and_lr_sched(
    args: ArgsBiSSLightClassificationMAELora,
    pretrain_config: ArgsPretextDefaults,
    parameter_groups: (
        List[Dict[str, Iterator[torch.nn.Parameter] | float]]
        | List[Dict[str, List[torch.nn.Parameter] | float]]
    ),
) -> Tuple[torch.optim.Optimizer, CosineLRSchedulerWithWarmup]:

    assert all(
        ("params" in pg.keys() and "weight_decay" in pg.keys())
        for pg in parameter_groups
    )

    optim_kwargs = {}

    if args.p_momentum is not None:
        optim_kwargs["momentum"] = args.p_momentum
    elif hasattr(pretrain_config, "momentum"):
        optim_kwargs["momentum"] = getattr(pretrain_config, "momentum")

    optim_kwargs["betas"] = list(ADAM_BETAS_DEFAULTS)

    for i, kwarg in enumerate(["beta1", "beta2"]):
        if getattr(args, f"p_{kwarg}") is not None:
            optim_kwargs["betas"][i] = getattr(args, f"p_{kwarg}")
        elif hasattr(pretrain_config, kwarg):
            optim_kwargs["betas"][i] = getattr(pretrain_config, kwarg)
        # In case not specified, the optimizer will use its default values for these hyperparameters.

    optim_kwargs["betas"] = tuple(optim_kwargs["betas"])

    optimizer = get_optimizer(
        args.p_optimizer or pretrain_config.optimizer,
        parameter_groups,
        lr=args.p_lr or pretrain_config.lr,
        **optim_kwargs,
    )

    lr_scheduler = CosineLRSchedulerWithWarmup(
        optimizer=optimizer,
        n_epochs=0,  # args.epochs * args.p_iter_epoch + add_epochs,
        len_loader=0,
        warmup_epochs=0,
        end_lrs=1e-6,
    )

    # As the warmup and remainder of training have different number of steps,
    # we need to set the total_steps and warmup_steps manually
    lr_scheduler.total_steps = args.lower_num_iter * args.epochs
    lr_scheduler.warmup_steps = (
        args.lr_scheduler_warmup_epochs * args.lower_num_iter
        + 1  # +1 since we do step() before training starts
    )
    lr_scheduler.step()  # To set the initial lr correctly

    return optimizer, lr_scheduler


def get_upper_downstream_optimizer_and_lr_sched(
    args: ArgsBiSSLightClassificationMAELora,
    parameter_groups: List[
        Dict[str, Iterator[torch.nn.Parameter] | List[torch.nn.Parameter] | float]
    ],
):

    assert all(
        ("params" in pg.keys() and "weight_decay" in pg.keys())
        for pg in parameter_groups
    )

    optimizer = get_optimizer(
        args.d_optimizer,
        parameter_groups,
        lr=args.d_lr,
        momentum=args.d_momentum,
        betas=(args.d_beta1, args.d_beta2),
    )

    lr_scheduler = CosineLRSchedulerWithWarmup(
        optimizer=optimizer,
        n_epochs=args.epochs,
        len_loader=args.upper_num_iter,
        warmup_epochs=args.lr_scheduler_warmup_epochs,
        end_lrs=1e-6,
    )

    return optimizer, lr_scheduler
