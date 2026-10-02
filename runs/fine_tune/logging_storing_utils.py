from typing import Dict, Literal, Optional
import yaml
import torch

from source.logging import BaseLogger


def log_stats(
    logger: BaseLogger,
    tr_stats: Dict[Literal["top1", "top5", "loss"], float],
    te_stats: Dict[Literal["top1", "top5", "loss"], float],
    lr: float,
    epoch: int,
    best_top1_acc: float,
    best_top5_acc: float,
    cuda_memory_peak=None,
):

    log_dict = {
        "train_ft/top1_acc": tr_stats["top1"],
        "train_ft/top5_acc": tr_stats["top5"],
        "train_ft/avgloss": tr_stats["loss"],
        "test_ft/top1_acc": te_stats["top1"],
        "test_ft/top5_acc": te_stats["top5"],
        "test_ft/avgloss": te_stats["loss"],
        "test_ft/best_top1_acc": best_top1_acc,
        "test_ft/best_top5_acc": best_top5_acc,
        "test_ft/lr": lr,
        "test_ft/epoch": epoch + 1,
    }
    if cuda_memory_peak is not None:
        log_dict.update({"test/cuda_memory_peak": cuda_memory_peak})
    logger.log(log_dict)


def get_state_dicts_to_store(
    model: torch.nn.Module,
    using_lora: bool = False,
    unfreeze_norm_layers: bool = False,
    store_backbone: bool = True,
):
    model.eval()

    if store_backbone:
        sd_bb = {k: v.cpu() for k, v in model.backbone.state_dict().items()}  # type: ignore

        if using_lora:
            incl_list = ["lora"]
            if unfreeze_norm_layers:
                incl_list += ["norm"]
            [
                sd_bb.pop(key)
                for key in list(sd_bb.keys())
                if not any(incl in key for incl in incl_list)
            ]
    else:
        sd_bb = "Frozen Backbone - Weights Not Updated During Fine-Tuning"

    sd_h = {k: v.cpu() for k, v in model.head.state_dict().items()}  # type: ignore

    return sd_bb, sd_h


def log_data_and_state_dicts(
    args,
    logger: BaseLogger,
    model: torch.nn.Module,
    storage_paths_dict: Optional[
        Dict[Literal["backbone_path", "head_path", "config_path"], str]
    ],
    tr_stats: Dict[Literal["top1", "top5", "loss"], float],
    te_stats: Dict[Literal["top1", "top5", "loss"], float],
    lr: float,
    epoch: int,
    best_top1_acc: float,
    best_top5_acc: float,
    best_acc_top1_prev: float,
    cuda_memory_peak=None,
):

    if args.rank != 0:
        return

    assert tr_stats.keys() == te_stats.keys() == {"top1", "top5", "loss"}
    log_stats(
        logger=logger,
        tr_stats=tr_stats,
        te_stats=te_stats,
        lr=lr,
        epoch=epoch,
        best_top1_acc=best_top1_acc,
        best_top5_acc=best_top5_acc,
        cuda_memory_peak=cuda_memory_peak,
    )

    if (args.model_root is not None) and (best_top1_acc - best_acc_top1_prev) > 0:
        assert storage_paths_dict is not None

        # Save model weights
        sd_bb, sd_h = get_state_dicts_to_store(
            model=model,
            using_lora=bool(getattr(args, "lora_rank", None)),
            unfreeze_norm_layers=bool(getattr(args, "unfreeze_norm_layers", False)),
            store_backbone=True,
        )
        torch.save(sd_bb, storage_paths_dict["backbone_path"])
        torch.save(sd_h, storage_paths_dict["head_path"])

        del sd_bb, sd_h

        # Save config file
        if hasattr(logger, "config"):
            config = logger.config  # type: ignore
            with open(storage_paths_dict["config_path"], "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(
                f"Model and config saved to {storage_paths_dict['backbone_path']}, {storage_paths_dict['head_path']}, and {storage_paths_dict['config_path']}."
            )
        else:
            print(
                f"Model saved to {storage_paths_dict['backbone_path']} and {storage_paths_dict['head_path']}. No config found in logger."
            )
