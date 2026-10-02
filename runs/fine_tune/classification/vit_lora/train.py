import json
import sys
from typing import Any, Dict, Optional

import torch
from ray.air import session

from source.augmentations.downstream import (
    DownstreamClassificationTestTransform,
    DownstreamClassificationTrainTransformViT,
)
from source.logging.console import ConsoleLogger
from source.logging.wandb import WandBLogger
from source.models.downstream.classification import DSClassifier
from source.models.misc import lora_wrapper
from source.models.vit import param_groups_lrd
from source.trainers import train_classifier
from source.types import ViTBackbones

from runs.dataloader import get_dataloader_and_sampler

from runs.fine_tune.misc import init_run
from runs.fine_tune.model import generate_storage_paths_ft
from runs.fine_tune.optimizer import get_optimizer_and_lr_sched
from runs.fine_tune.logging_storing_utils import log_data_and_state_dicts

from runs.fine_tune.classification.train import ft_eval_classifier
from runs.fine_tune.classification.vit_lora.config import ArgsFTClassificationViTLora


def train_classification_vit(
    hpo_config: Dict[str, Any],
    args: ArgsFTClassificationViTLora,
    backbone_arch: ViTBackbones,
    pretrained_bb_path: str,
    config_dict_to_upload: Optional[Dict[str, Any]] = None,
    pretrained_lora_path: Optional[str] = None,
    seed: Optional[int] = None,
    using_ray: bool = True,
):

    device = init_run(args, hpo_config, seed=seed, disable_print=False)

    if args.wandb_api_key is not None:
        logger = WandBLogger(
            project=args.run_type,
            config=args.as_dict() | (config_dict_to_upload or {}),
            rank=args.rank,
        )
    else:
        logger = ConsoleLogger(
            config=args.as_dict() | (config_dict_to_upload or {}),
            rank=args.rank,
        )

    print("(FT Classification) " + " ".join(sys.argv))
    print(json.dumps(dict(train_type="Console Args", data=" ".join(sys.argv))))
    print("")
    print(f"Device = {device}")

    assert args.img_size is not None
    assert args.img_crop_min_ratio is not None

    #### DATA IMPORT ####
    assert (
        args.dataset is not None
    ), "Dataset must be specified for downstream task. If use_hpo=0, specify dataset using the --dataset flag. Otherwise the ft config is corrupted."

    dataloader_train, sampler_train = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.dataset,
        split="train",
        batch_size=args.batch_size,
        transform=DownstreamClassificationTrainTransformViT(
            img_size=args.img_size,
            min_ratio=args.img_crop_min_ratio,
        ),
        img_size=args.img_size,
    )

    dataloader_val, sampler_val = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.dataset,
        split="val",
        batch_size=args.batch_size,
        transform=DownstreamClassificationTestTransform(
            img_size=args.img_size,
        ),
        img_size=args.img_size,
    )

    n_classes = len(dataloader_train.dataset.classes)  # type: ignore

    ### MODEL IMPORT ###
    model = DSClassifier(
        backbone_arch=backbone_arch,
        n_classes=n_classes,
        img_size=args.img_size,
        vit_drop_path_rate=args.vit_drop_path_rate,
    )
    pretrained_bb_sd = torch.load(pretrained_bb_path, map_location="cpu")

    # Quickfix to load the official MAE weights in a manner that is compatible with the DSClassifier model.
    # The official MAE weights are stored in a state dict with a "model" key, while the DSClassifier model expects the weights to be stored in a state dict without a "model" key.
    if "model" in pretrained_bb_sd.keys():
        pretrained_bb_sd = pretrained_bb_sd["model"]
    else:
        raise ValueError(
            f"Unsupported pretrained_bb_path: {pretrained_bb_path}. The state dict does not contain a 'model' or 'encoder' key."
        )

    load_bb_result = model.backbone.load_state_dict(pretrained_bb_sd, strict=False)

    print("Pretrained backbone loaded.")
    del pretrained_bb_sd

    assert (
        len(load_bb_result.missing_keys) == 0
    ), f"Missing keys when loading pretrained backbone: {load_bb_result.missing_keys}"

    if args.lora_rank is not None:
        model.backbone = lora_wrapper(  # type: ignore
            model=model.backbone,
            r=args.lora_rank,
            dropout=args.lora_dropout,
            unfreeze_norm_layers=bool(args.unfreeze_norm_layers),
        )

        if pretrained_lora_path is not None:
            lora_sd = torch.load(pretrained_lora_path, map_location="cpu")
            load_lora_result = model.backbone.load_state_dict(lora_sd, strict=False)
            incl_keys = ["lora"]
            if bool(args.unfreeze_norm_layers):
                incl_keys += ["norm"]
            missing_lora_keys = [
                key
                for key in load_lora_result.missing_keys
                if any(incl in key for incl in incl_keys)
            ]
            assert (
                len(missing_lora_keys) == 0
            ), f"Missing keys when loading pretrained LoRA weights: {load_lora_result.missing_keys}"

            del lora_sd

    torch.cuda.empty_cache()

    if args.use_grad_checkpointing:
        model.backbone.set_grad_checkpointing(True)

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)

    model.backbone = model.module.backbone
    model.head = model.module.head

    ### TRAINING SETUP ###
    loss_fn = torch.nn.CrossEntropyLoss()
    assert args.wd is not None

    param_groups = param_groups_lrd(
        model.module.backbone,
        weight_decay=args.wd,
        no_weight_decay_list=list(model.module.backbone.no_weight_decay())
        + (
            [
                module_name
                for module_name, _ in model.module.backbone.named_parameters()
                if "norm" in module_name
            ]
        ),
        layer_decay=(
            1.0 if args.lora_rank is not None else 0.75
        ),  # Do not apply layer decay when using LoRA, as only the adapter layers are being trained.
    ) + [
        dict(
            params=model.module.head.parameters(),
            weight_decay=args.wd,
        )
    ]

    optimizer, lr_scheduler = get_optimizer_and_lr_sched(
        args=args,
        parameter_groups=param_groups,  # type: ignore
        len_loader=len(dataloader_train),
        d_lr_sched_warmup_epochs=args.lr_scheduler_warmup_epochs,
    )

    if args.model_root is not None:
        storage_paths_dict = generate_storage_paths_ft(
            model_root=args.model_root,
            run_type=args.run_type,
            backbone_arch=backbone_arch,
            downstream_dataset=args.dataset,
            lora_rank=args.lora_rank,
            seed=args.seed,
        )
    else:
        storage_paths_dict = None

    best_acc_top1 = 0.0
    best_acc_top5 = 0.0
    cuda_memory_peak_max = 0

    print("'Downstream Fine-Tuning (Classification)")

    for epoch in range(int(args.epochs)):

        sampler_train.set_epoch(epoch)

        print("")
        print(f"Epoch {epoch+1} / {args.epochs}\n-------------------------------")

        torch.cuda.reset_peak_memory_stats()

        # Adapt backbone from pretext into downstream model
        tr_stats = train_classifier(
            model=model,
            loss_fn=loss_fn,
            dataloader=dataloader_train,
            optimizer=optimizer,
            device=device,
            lr_scheduler=lr_scheduler,
            use_amp=bool(args.use_amp),
            return_loss_avg=True,
            eval_accuracy=True if "voc07" not in args.dataset else False,
        )

        local_peak = torch.tensor(
            torch.cuda.max_memory_allocated(),
            device=torch.cuda.current_device(),
            dtype=torch.float64,
        )
        cuda_memory_peak = torch.tensor(
            local_peak,
            device=torch.cuda.current_device(),
            dtype=torch.float64,
        )

        torch.distributed.all_reduce(
            cuda_memory_peak, op=torch.distributed.ReduceOp.SUM
        )

        if args.rank == 0:
            cuda_memory_peak_max = max(cuda_memory_peak_max, cuda_memory_peak.item())

        print("")

        sampler_val.set_epoch(epoch)
        _, te_stats = ft_eval_classifier(
            args=args,
            model=model,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            loss_fn=loss_fn,
            device=device,
            eval_train=False,
            use_amp=bool(args.use_amp),
        )

        best_acc_top1_prev = best_acc_top1
        best_acc_top1 = max(best_acc_top1, te_stats["top1"])
        best_acc_top5 = max(best_acc_top5, te_stats["top5"])

        if using_ray:
            session.report({"loss": te_stats["loss"], "accuracy": te_stats["top1"]})

        log_data_and_state_dicts(
            args=args,
            logger=logger,
            model=model,
            storage_paths_dict=storage_paths_dict,
            tr_stats=tr_stats,  # type: ignore
            te_stats=te_stats,
            lr=optimizer.param_groups[0]["lr"],
            epoch=epoch,
            best_top1_acc=best_acc_top1,
            best_top5_acc=best_acc_top5,
            best_acc_top1_prev=best_acc_top1_prev,
            cuda_memory_peak=cuda_memory_peak_max,
        )

        cuda_memory_peak_max = 0

    logger.close()

    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
