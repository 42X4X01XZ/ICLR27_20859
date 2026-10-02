from typing import Optional, Tuple

import numpy as np
import torch
from timm.utils.metrics import AverageMeter, accuracy
from torch.utils.data import DataLoader


def eval_classifier_top1_and_top5(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    label: str = "Test Performance",
    non_blocking: bool = True,
    use_amp: bool = False,
) -> Tuple[float, float, float]:
    # Fmi: Its very important that the model is in eval mode if batchnorms are included...
    model.eval()
    test_loss = 0

    top1 = AverageMeter()
    top5 = AverageMeter()

    assert (
        len(dataloader) > 0
    ), "Dataloader is empty. Please check the dataset and dataloader configuration."

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=non_blocking), y.to(
                device, non_blocking=non_blocking
            )
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(X)
            else:
                pred = model(X)
            test_loss += loss_fn(pred, y).item()
            acc1, acc5 = accuracy(pred, y, topk=(1, 5))
            top1.update(acc1.item(), X.size(0))
            top5.update(acc5.item(), X.size(0))

        test_loss /= len(dataloader)

    del X, y, pred, acc1, acc5  # type: ignore

    if torch.distributed.is_initialized():
        metrics = torch.tensor(
            [
                top1.sum,
                top1.count,
                top5.sum,
                top5.count,
                test_loss,
            ],
            device=device,
        )
        torch.cuda.synchronize()
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

        top1 = metrics[0].item() / metrics[1].item()
        top5 = metrics[2].item() / metrics[3].item()
        test_loss = metrics[4].item() / torch.distributed.get_world_size()
    else:
        top1, top5 = top1.avg, top5.avg

    torch.cuda.empty_cache()

    print(
        f"{label}: \n Top1 Acc: {top1:.2f}%, Top5 Acc: {top5:.2f}%, Avg loss: {test_loss:.5f} \n"
    )
    return top1, top5, test_loss


def eval_classifier_mAP(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label: str = "Test Performance",
    non_blocking: bool = True,
    use_amp: bool = False,
    loss_fn: Optional[torch.nn.Module] = None,
) -> Tuple[float, float]:
    model.eval()

    test_loss = 0.0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=non_blocking), y.to(
                device, non_blocking=non_blocking
            )

            # Forward pass to get logits
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(X)
            else:
                logits = model(X)

            if loss_fn is not None:
                test_loss += loss_fn(logits, y).item()

            all_logits.append(logits.float())
            all_targets.append(y)

        test_loss /= len(dataloader)

    # Concatenate all results from all batches (ensure that these are all Tensors)
    all_logits = torch.cat(all_logits)  # Now a single Tensor
    all_targets = torch.cat(all_targets)  # Now a single Tensor

    # Convert Tensors to NumPy arrays for easier processing
    all_logits = all_logits.cpu().numpy()
    all_targets = all_targets.cpu().numpy()

    # Compute mAP for each class
    num_classes = all_logits.shape[1]
    aps = []

    for cls in range(num_classes):
        # Get ground truth and logits for this class
        gt = all_targets[:, cls]
        scores = all_logits[:, cls]

        # Check if there are any positive samples for this class
        if np.sum(gt) == 0:
            # No positive samples in ground truth, skip this class or handle as needed
            print(
                f"Warning: No positive samples for class {cls}. Skipping mAP calculation for this class."
            )
            continue

        # Sort by logits directly (higher logits indicate higher confidence in that class)
        sorted_indices = np.argsort(-scores)
        sorted_gt = gt[sorted_indices]

        # Compute true positives and false positives
        tp = np.cumsum(sorted_gt)
        fp = np.cumsum(1 - sorted_gt)

        # Compute precision and recall
        recalls = tp / np.sum(gt)
        precisions = tp / (tp + fp)

        # 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            precisions_at_recall = precisions[recalls >= t]
            p = np.max(precisions_at_recall) if precisions_at_recall.size > 0 else 0
            ap += p / 11
        aps.append(ap)

    # Compute the mean Average Precision (mAP)
    if len(aps) > 0:
        mAP = 100 * torch.tensor(aps, dtype=torch.float32, device=device).mean()
    else:
        mAP = torch.tensor(0.0, dtype=torch.float32, device=device)

    if torch.distributed.is_initialized():
        torch.cuda.synchronize()
        mAP = mAP.clone()
        torch.distributed.all_reduce(mAP, op=torch.distributed.ReduceOp.SUM)
        mAP /= torch.distributed.get_world_size()

        if loss_fn is not None:
            test_loss_tensor = torch.tensor(test_loss, device=device)
            torch.distributed.all_reduce(
                test_loss_tensor, op=torch.distributed.ReduceOp.SUM
            )
            test_loss = test_loss_tensor.item() / torch.distributed.get_world_size()

    print(f"{label}: \n mAP: {mAP.item():.4f} \n")

    return float(mAP.item()), test_loss
