from typing import Any, Dict, Optional

import torch

from runs.pretext.config import ArgsPretextDefaults
from runs.bisslight.config import ArgsBiSSLDefaults
from runs.fine_tune.config import ArgsFineTuningDefaults
from runs.distributed import set_seed, setup_for_distributed


def init_run(
    args: ArgsFineTuningDefaults,
    hpo_config: Dict[str, Any],
    seed: Optional[int] = None,
    disable_print: bool = True,
) -> torch.device:

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)

    if args.world_size > 1:
        torch.distributed.barrier()

    device = torch.device(args.gpu)

    if disable_print:
        # Disables all printing, as the HPO will print plenty :)
        setup_for_distributed(False)

    if seed is not None:
        args.seed = seed
    else:
        set_seed(args, force_generate_new_seed=True)

    # HPO Config Import
    for key, val in hpo_config.items():
        if bool(args.use_hpo) or vars(args)[key] is None:
            vars(args)[key] = val

    return device


def update_args_postbissl(
    args: ArgsFineTuningDefaults,
    bissl_config: ArgsBiSSLDefaults,
    pretext_config: ArgsPretextDefaults,
) -> None:
    """Updates the fine-tuning config with relevant hyperparameters from the BiSSLight/BiSSL config.

    Args:
        args (ArgsFineTuningDefaults): Arguments for the fine-tuning run.
        bissl_config (ArgsBiSSLDefaults): Arguments for the BiSSL run.
    """
    if args.dataset is not None:
        print(
            "WARNING: Dataset should be overrided with BiSSL config. "
            + "This is typically not intended behaviour, and is mainly an experimental feature."
            + " Will likely raise an error if no manual intervention has been done on the"
            + " pretrained model paths."
        )

    args.dataset = args.dataset or bissl_config.d_dataset
    args.batch_size = args.batch_size or bissl_config.d_batch_size
    args.optimizer = args.optimizer or bissl_config.d_optimizer
    args.momentum = args.momentum or bissl_config.d_momentum
    args.beta1 = args.beta1 or bissl_config.d_beta1
    args.beta2 = args.beta2 or bissl_config.d_beta2

    if hasattr(args, "img_size"):
        args.img_size = (  # type: ignore
            args.img_size or bissl_config.d_img_size or pretext_config.img_size  # type: ignore
        )
        assert args.img_size is not None  # type: ignore

    if hasattr(args, "img_crop_min_ratio"):
        args.img_crop_min_ratio = (  # type: ignore
            args.img_crop_min_ratio  # type: ignore
            or bissl_config.d_img_crop_min_ratio  # type: ignore
            or pretext_config.img_crop_min_ratio
        )
        assert args.img_crop_min_ratio is not None  # type: ignore

    # LoRA Specific
    if hasattr(args, "lora_rank"):
        args.lora_rank = args.lora_rank if args.lora_rank is not None else bissl_config.lora_rank  # type: ignore
        if args.lora_rank is None:  # type: ignore
            print(
                "WARNING: LoRA rank is not specified in either the fine-tuning config or the BiSSL config. "
                + "Full fine-tuning will be performed instead of LoRA fine-tuning. "
            )
    if hasattr(args, "unfreeze_norm_layers"):
        args.unfreeze_norm_layers = args.unfreeze_norm_layers if args.unfreeze_norm_layers is not None else bissl_config.unfreeze_norm_layers  # type: ignore
        assert args.unfreeze_norm_layers is not None  # type: ignore

    # These are critical hyperparameters that must be specified either in the fine-tuning config or the BiSSL config.
    assert all(
        arg is not None
        for arg in (
            args.batch_size,
            args.optimizer,
            args.momentum,
            args.beta1,
            args.beta2,
        )
    )


def get_pretrain_config_dict(
    pretext_config: ArgsPretextDefaults,
    bissl_config: Optional[ArgsBiSSLDefaults] = None,
) -> Dict[str, Any]:

    pretrain_config_dict = dict(PT=pretext_config.as_dict())
    if bissl_config is not None:
        pretrain_config_dict.update(dict(BiSSL=bissl_config.as_dict()))
    return pretrain_config_dict
