import os
from typing import Literal, Dict, Optional
from source.types import BackboneArchs, DownstreamDatasets


def generate_storage_paths_ft(
    model_root: str,
    run_type: Literal["ft-post-pretext", "ft-post-bissl", "ft-post-bisslight"],
    backbone_arch: BackboneArchs,  # type: ignore
    downstream_dataset: DownstreamDatasets,
    lora_rank: Optional[int] = None,
    seed: Optional[int] = None,
    make_dirs: bool = True,
) -> Dict[Literal["backbone_path", "head_path", "config_path"], str]:

    # Build filename prefix
    filename_parts = [
        run_type,  # Run type ("ft-post-pretext" (fine-tuning the pretrained backbone directly), "ft-post-bissl"/"ft-post-bisslight" (fine-tuning the bissl/bisslight pretrained model), etc.)
        backbone_arch,  # Model architecture (e.g., "vit_huge")
        downstream_dataset,  # Downstream dataset (e.g., "cub200")
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
    model_head_path = os.path.join(model_root, f"{filename_prefix}_head.pth.tar")
    config_path = os.path.join(model_root, f"{filename_prefix}_config.yaml")

    # Create the directory if it doesn't exist
    if make_dirs:
        os.makedirs(model_root, exist_ok=True)

    storage_paths_dict: Dict[
        Literal["backbone_path", "head_path", "config_path"], str
    ] = {
        "backbone_path": model_backbone_path,
        "head_path": model_head_path,
        "config_path": config_path,
    }

    return storage_paths_dict
