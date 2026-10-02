from typing import Optional
import contextlib

import torch
from torch.utils.data import DataLoader
from source.schedulers import CosineLRSchedulerWithWarmup, ConstantLR
from timm.utils.metrics import AverageMeter, accuracy


def train_classifier(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lr_scheduler: Optional[CosineLRSchedulerWithWarmup | ConstantLR] = None,
    return_loss_avg: bool = False,
    use_amp: bool = False,
    eval_accuracy: bool = False,
    grad_accum_steps: int = 1,
):
    loss_avg = 0
    model.train()

    if eval_accuracy:
        top1 = AverageMeter()
        top5 = AverageMeter()

    n_batches = len(dataloader)
    optimizer.zero_grad(set_to_none=True if grad_accum_steps == 1 else False)

    for batch_idx, (imgs, labels) in enumerate(dataloader):
        imgs, labels = (
            imgs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )

        is_last_batch = batch_idx == n_batches - 1
        do_step = ((batch_idx + 1) % grad_accum_steps == 0) or is_last_batch

        # Skip DDP gradient all-reduce on intermediate accumulation steps
        sync_ctx = contextlib.nullcontext() if do_step else model.no_sync()  # type: ignore

        with sync_ctx:
            if use_amp:
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):  # type : ignore
                    out = model(imgs)
                    loss = loss_fn(out, labels)
            else:
                out = model(imgs)
                loss = loss_fn(out, labels)

            loss.backward()  # type: ignore

        if return_loss_avg or eval_accuracy:
            loss_avg += loss.item()

        if eval_accuracy:
            with torch.no_grad():
                acc1, acc5 = accuracy(out, labels, topk=(1, 5))
                top1.update(acc1.item(), imgs.size(0))  # type: ignore
                top5.update(acc5.item(), imgs.size(0))  # type: ignore
            del acc1, acc5

        del out, imgs, labels, loss

        if do_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True if grad_accum_steps == 1 else False)

            if lr_scheduler is not None:
                lr_scheduler.step()

    optimizer.zero_grad(set_to_none=True)

    if return_loss_avg or eval_accuracy:
        loss_avg /= len(dataloader)

    if eval_accuracy:
        if torch.distributed.is_initialized():
            metrics = torch.tensor(
                [
                    top1.sum,  # type: ignore
                    top1.count,  # type: ignore
                    top5.sum,  # type: ignore
                    top5.count,  # type: ignore
                    loss_avg,
                ],
                device=device,
            )
            torch.cuda.synchronize()
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

            top1 = metrics[0].item() / metrics[1].item()
            top5 = metrics[2].item() / metrics[3].item()
            loss_avg = metrics[4].item() / torch.distributed.get_world_size()
        else:
            top1, top5 = top1.avg, top5.avg  # type: ignore

        return {
            "top1": top1,
            "top5": top5,
            "loss": loss_avg,
        }

    print(
        f"Avg loss over batch no. 1-{len(dataloader)} / {len(dataloader)}: {loss_avg:>5f}",
    )

    torch.cuda.empty_cache()

    if return_loss_avg:
        return {
            "top1": 0.0,
            "top5": 0.0,
            "loss": loss_avg,
        }
