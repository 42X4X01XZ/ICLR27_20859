from typing import Optional, Dict, Literal

import torch
import yaml

from source.logging.base import BaseLogger


def log_stats(
    logger: BaseLogger,
    top1_dbb: float,
    top5_dbb: float,
    loss_dbb: float,
    epoch: int,
    best_top1_acc: float,
    best_top5_acc: float,
    top1_pbb: Optional[float] = None,
    best_top1_acc_pbb: Optional[float] = None,
    top5_pbb: Optional[float] = None,
    best_top5_acc_pbb: Optional[float] = None,
    loss_pbb: Optional[float] = None,
    cuda_memory_peak: Optional[float] = None,
):
    log_dict = {
        "test/top1_acc_test_dbb": top1_dbb,
        "test/top5_acc_test_dbb": top5_dbb,
        "test/best_top1_acc_test_dbb": best_top1_acc,
        "test/best_top5_acc_test_dbb": best_top5_acc,
        "test/avgloss_test_dbb": loss_dbb,
        "test/epoch": epoch + 1,
    }

    if top1_pbb is not None and top5_pbb is not None and loss_pbb is not None:
        log_dict.update(
            {
                "test/top1_acc_test_pbb": top1_pbb,
                "test/best_top1_acc_test_pbb": best_top1_acc_pbb,
                "test/top5_acc_test_pbb": top5_pbb,
                "test/best_top5_acc_test_pbb": best_top5_acc_pbb,
                "test/avgloss_test_pbb": loss_pbb,
            }
        )
    if cuda_memory_peak is not None:
        log_dict.update({"test/cuda_memory_peak": cuda_memory_peak})
    logger.log(log_dict)


def get_bb_state_dict_to_store(
    model: torch.nn.Module,
    using_lora: bool = False,
    unfreeze_norm_layers: bool = False,
):
    model.eval()

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

    return sd_bb


def log_data_and_state_dicts(
    args,
    logger: BaseLogger,
    model_p: torch.nn.Module,
    storage_paths_dict: Optional[Dict[Literal["backbone_path", "config_path"], str]],
    top1_dbb: float,
    top5_dbb: float,
    loss_dbb: float,
    epoch: int,
    best_top1_acc: float,
    best_top5_acc: float,
    top1_pbb: Optional[float] = None,
    best_top1_acc_pbb: Optional[float] = None,
    top5_pbb: Optional[float] = None,
    best_top5_acc_pbb: Optional[float] = None,
    loss_pbb: Optional[float] = None,
    cuda_memory_peak: Optional[float] = None,
):

    if args.rank != 0:
        return

    log_stats(
        logger=logger,
        top1_dbb=top1_dbb,
        top5_dbb=top5_dbb,
        loss_dbb=loss_dbb,
        epoch=epoch,
        best_top1_acc=best_top1_acc,
        best_top5_acc=best_top5_acc,
        top1_pbb=top1_pbb,
        best_top1_acc_pbb=best_top1_acc_pbb,
        top5_pbb=top5_pbb,
        best_top5_acc_pbb=best_top5_acc_pbb,
        loss_pbb=loss_pbb,
        cuda_memory_peak=cuda_memory_peak,
    )

    if args.model_root is not None:
        assert storage_paths_dict is not None
        model_p.eval()
        state_dict_p_backbone = get_bb_state_dict_to_store(
            model=model_p,
            using_lora=bool(getattr(args, "lora_rank", None)),
            unfreeze_norm_layers=bool(getattr(args, "unfreeze_norm_layers", False)),
        )

        torch.save(state_dict_p_backbone, storage_paths_dict["backbone_path"])

        del state_dict_p_backbone

        if hasattr(logger, "config"):
            config = logger.config  # type: ignore
            with open(storage_paths_dict["config_path"], "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print(
                f"Model and config saved to {storage_paths_dict['backbone_path']}, and {storage_paths_dict['config_path']}."
            )
        else:
            print(
                f"Model saved to {storage_paths_dict['backbone_path']}. No config found in logger."
            )

        print("")
