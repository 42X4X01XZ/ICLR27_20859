import os
from typing import Callable, Dict, Optional, Tuple, Literal
import torch

from source.types import (
    BackboneArchs,
    DownstreamDatasets,
    DownstreamTaskTypes,
    PretextTaskTypes,
)


def wrap_model_for_ddp(
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    find_unused_parameters: bool = False,
    device_id: Optional[int] = None,
    static_graph: bool = False,
) -> torch.nn.Module:
    model = model if device is None else model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=find_unused_parameters,
        device_ids=[device_id] if device_id is not None else None,
        static_graph=static_graph,
    )

    model.backbone = model.module.backbone
    model.head = model.module.head

    torch.distributed.barrier()

    return model


def generate_storage_paths_bissl(
    model_root: str,
    run_type: Literal["bissl", "bisslight"],
    pretext_task: PretextTaskTypes,
    downstream_task: DownstreamTaskTypes,
    backbone_arch: BackboneArchs,  # type: ignore
    downstream_dataset: DownstreamDatasets,
    lora_rank: Optional[int] = None,
    seed: Optional[int] = None,
    make_dirs: bool = True,
) -> Dict[Literal["backbone_path", "config_path"], str]:

    # Build filename prefix
    filename_parts = [
        run_type,  # Run type (bissl, bisslight, fine-tuning the mae pretrained model directly, etc.)
        backbone_arch,  # Model architecture (e.g., "vit_huge")
        f"l-{pretext_task}",  # Pretext task (e.g., "mae")
        f"u-{downstream_task}-{downstream_dataset}",  # Downstream task (e.g., "classification")
    ]

    # Add LoRA rank (if used)
    if lora_rank is not None:
        filename_parts.append(f"lora-r{lora_rank}")

    # Add seed (used as unique run ID)
    if seed is not None:
        filename_parts.append(f"seed{seed}")

    # Join parts to form the filename prefix
    filename_prefix = "_".join(filename_parts)

    # Define paths
    model_backbone_path = os.path.join(
        model_root, f"{filename_prefix}_backbone.pth.tar"
    )
    config_path = os.path.join(model_root, f"{filename_prefix}_config.yaml")

    # Create the directory if it doesn't exist
    if make_dirs:
        os.makedirs(model_root, exist_ok=True)

    storage_paths_dict: Dict[Literal["backbone_path", "config_path"], str] = {
        "backbone_path": model_backbone_path,
        "config_path": config_path,
    }

    return storage_paths_dict


def get_lower_input_processor_single(
    device: torch.device,
) -> Callable:
    """
    Returns a function that processes a tuple of input data and labels respectively, and returns the processed input data.
    Compatible with pretext tasks that operate on a single image, such as MAE.
    """

    def return_fn(x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        img, _ = x
        return img.to(device, non_blocking=True)

    return return_fn
