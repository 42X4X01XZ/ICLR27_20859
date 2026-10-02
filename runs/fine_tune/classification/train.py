from typing import Dict, Literal, Tuple

import torch
from torch.utils.data import DataLoader

from source.evals import (
    eval_classifier_mAP,
    eval_classifier_top1_and_top5,
)

from runs.fine_tune.classification.config import ArgsFTClassification


def _ft_eval_classifier_single(
    args: ArgsFTClassification,
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    label: str = "Train Performance",
    non_blocking: bool = True,
    use_amp: bool = False,
) -> Dict[Literal["top1", "top5", "loss"], float]:
    if args.dataset == "voc07":
        top1, loss = eval_classifier_mAP(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            device=device,
            label=label + " (mAP)",
            non_blocking=non_blocking,
            use_amp=use_amp,
        )

        top5 = 0.0
    else:
        top1, top5, loss = eval_classifier_top1_and_top5(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            device=device,
            label=label + " (Top1 and Top5)",
            non_blocking=non_blocking,
            use_amp=use_amp,
        )

    stats: Dict[Literal["top1", "top5", "loss"], float] = {
        "top1": top1,
        "top5": top5,
        "loss": loss,
    }
    return stats


def ft_eval_classifier(
    args: ArgsFTClassification,
    model: torch.nn.Module,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    label: str = "Train Performance",
    non_blocking: bool = True,
    eval_train: bool = True,
    use_amp: bool = False,
) -> Tuple[
    Dict[Literal["top1", "top5", "loss"], float],
    Dict[Literal["top1", "top5", "loss"], float],
]:
    if eval_train:
        train_stats: Dict[Literal["top1", "top5", "loss"], float] = (
            _ft_eval_classifier_single(
                args=args,
                model=model,
                dataloader=dataloader_train,
                loss_fn=loss_fn,
                device=device,
                label=label,
                non_blocking=non_blocking,
                use_amp=use_amp,
            )
        )
        torch.cuda.empty_cache()
    else:
        train_stats = {"top1": 0.0, "top5": 0.0, "loss": 0.0}

    val_stats: Dict[Literal["top1", "top5", "loss"], float] = (
        _ft_eval_classifier_single(
            args=args,
            model=model,
            dataloader=dataloader_val,
            loss_fn=loss_fn,
            device=device,
            label="Validation Performance",
            non_blocking=non_blocking,
            use_amp=use_amp,
        )
    )
    torch.cuda.empty_cache()

    return train_stats, val_stats
