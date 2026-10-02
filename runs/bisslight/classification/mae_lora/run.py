from copy import deepcopy
import json
import sys
from typing import get_args

import torch
import torch.distributed as dist

from source.bissl.ij_solver_mfac import IJGradCalcMFACLW
from source.types import PretrainDatasets, ViTBackbones
from source.augmentations.downstream import (
    DownstreamClassificationTestTransform,
    DownstreamClassificationTrainTransformViT,
)
from source.augmentations.pretext import MAEPretextTransform
from source.datasets import GetData
from source.logging.console import ConsoleLogger
from source.logging.wandb import WandBLogger


from source.models.downstream.classification import DSClassifier
from source.models.misc import get_lora_layerwise_param_group_idxs, lora_wrapper
from source.models.pretext.mae import get_mae_model
from source.models.vit import add_weight_decay, param_groups_lrd

from runs.pretext.mae.config import ArgsPretextMAEOfficial

from runs.dataloader import get_dataloader_and_sampler
from runs.distributed import init_dist

from runs.bisslight.model import (
    generate_storage_paths_bissl,
    get_lower_input_processor_single,
    wrap_model_for_ddp,
)
from runs.bisslight.optimizer import (
    get_lower_pretext_optimizer_and_lr_sched,
    get_upper_downstream_optimizer_and_lr_sched,
)
from runs.bisslight.train import log_data_and_state_dicts

from runs.bisslight.classification.train import BiSSLightClassification
from runs.bisslight.classification.linear_warmup import ClassifierLinearWarmupTrainer
from runs.bisslight.classification.mae_lora.config import (
    ArgsBiSSLightClassificationMAELora,
)
from runs.bisslight.classification.misc import tie_models
from runs.bisslight.classification.model import get_upper_input_processor
from runs.bisslight.classification.train import eval_fn

if __name__ == "__main__":

    args = ArgsBiSSLightClassificationMAELora().parse_args()

    ### Distributed Setup ###
    device = init_dist(args)

    ### Fetch Pretrain Config and Init Wandb ###
    pretrain_config = ArgsPretextMAEOfficial()

    match args.pretrained_backbone_path:
        case s if s.split("/")[-1] == "mae_pretrain_vit_base_full.pth":
            backbone_arch = "vit_base_patch16"
        case s if s.split("/")[-1] == "mae_pretrain_vit_large_full.pth":
            backbone_arch = "vit_large_patch16"
        case s if s.split("/")[-1] == "mae_pretrain_vit_huge_full.pth":
            backbone_arch = "vit_huge_patch14"
        case _:
            raise ValueError(
                f"Unsupported pretrained_backbone_path: {args.pretrained_backbone_path}"
                + "Dir has to reference on of the following files (the official mae pretrained checkpoints): 'mae_pretrain_vit_base_full.pth', 'mae_pretrain_vit_large_full.pth', 'mae_pretrain_vit_huge_full.pth'"
            )
    # Official values read off from https://arxiv.org/abs/2111.06377
    pretrain_config.from_dict(
        {"dataset_root": args.dataset_root, "backbone_arch": backbone_arch}
    )

    assert pretrain_config.backbone_arch in get_args(ViTBackbones)

    if args.wandb_api_key is not None:
        logger = WandBLogger(
            project=args.run_type,
            config=args.as_dict() | dict(PT=pretrain_config.as_dict()),
            rank=args.rank,
        )
    else:
        logger = ConsoleLogger(
            config=args.as_dict() | dict(PT=pretrain_config.as_dict()),
            rank=args.rank,
        )

    if args.model_root is not None:
        storage_paths_dict = generate_storage_paths_bissl(
            model_root=args.model_root,
            run_type=args.run_type,
            backbone_arch=pretrain_config.backbone_arch,
            pretext_task=pretrain_config.pretext_task,
            downstream_task=args.downstream_task,
            downstream_dataset=args.d_dataset,
            lora_rank=args.lora_rank,
            seed=args.seed,
        )
    else:
        print("Model root not specified. Not saving model checkpoints.")
        print("")
        storage_paths_dict = None

    print("(BiSSLight) " + " ".join(sys.argv))

    print(json.dumps(dict(train_type="Console Args", data=" ".join(sys.argv))))

    print("")
    print(f"Device = {device}")

    #### DATA IMPORT ####
    get_data = GetData(args.dataset_root, bool(args.download))

    dataloader_d_train, sampler_d_train = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.d_dataset,
        split="train",
        batch_size=args.d_batch_size,
        transform=DownstreamClassificationTrainTransformViT(
            img_size=args.d_img_size or pretrain_config.img_size,
            min_ratio=args.d_img_crop_min_ratio or pretrain_config.img_crop_min_ratio,
        ),
        drop_last=True,
        img_size=args.d_img_size or pretrain_config.img_size,
    )
    dataloader_d_val, sampler_d_val = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.d_dataset,
        split="val",
        batch_size=args.d_batch_size,
        transform=DownstreamClassificationTestTransform(
            img_size=args.d_img_size or pretrain_config.img_size,
        ),
        img_size=args.d_img_size or pretrain_config.img_size,
    )
    n_classes = len(dataloader_d_train.dataset.classes)  # type: ignore

    print("Using downstream dataset: " + args.d_dataset)

    ### MODEL IMPORT ###
    model_p = get_mae_model(
        pretrain_config.backbone_arch,
        img_size=args.p_img_size or pretrain_config.img_size,
        norm_pix_loss=bool(args.mae_norm_pix_loss),
        drop_path_rate=args.p_vit_drop_path_rate,
        decoder_drop_path_rate=args.p_decoder_drop_path_rate,
    )

    sd_pretrained = torch.load(args.pretrained_backbone_path)["model"]

    load_res_bb = model_p.backbone.load_state_dict(sd_pretrained, strict=False)
    assert (
        len(load_res_bb.missing_keys) == 0
    ), f"Missing keys when loading pretrained model: {load_res_bb.missing_keys}"

    del sd_pretrained

    model_d = DSClassifier(
        pretrain_config.backbone_arch,
        n_classes=n_classes,
        img_size=args.d_img_size or pretrain_config.img_size,
        vit_drop_path_rate=args.d_vit_drop_path_rate,
    )

    if args.lora_rank is not None:
        model_d.backbone = lora_wrapper(  # type: ignore
            model=model_d.backbone,
            r=args.lora_rank,
            dropout=args.lora_dropout,
            unfreeze_norm_layers=bool(args.unfreeze_norm_layers),
        )
        model_p.backbone = lora_wrapper(  # type: ignore
            model=model_p.backbone,
            r=args.lora_rank,
            dropout=args.lora_dropout,
            unfreeze_norm_layers=bool(args.unfreeze_norm_layers),
        )
        if args.p_apply_lora_on_head:
            model_p.head = lora_wrapper(  # type: ignore
                model=model_p.head,
                r=args.lora_rank,
                dropout=args.lora_dropout,
                unfreeze_norm_layers=bool(args.unfreeze_norm_layers),
            )

    # Since we want the LoRA params to also initialize with identical weights, we load the state dict again.
    model_d.backbone.load_state_dict(model_p.backbone.state_dict())

    #### DOWNSTREAM HEAD WARMUP  ###
    if args.d_head_warmup_epochs > 0:
        model_d = wrap_model_for_ddp(
            model_d,
            device,
            find_unused_parameters=True,
        )
        lw_trainer = ClassifierLinearWarmupTrainer(
            args=args,
            logger=logger,
            model=model_d,
            dataloader_train=dataloader_d_train,
            sampler_train=sampler_d_train,
            dataloader_val=dataloader_d_val,
            sampler_val=sampler_d_val,
            device=device,
            use_amp=args.use_amp,
            eval_accuracy=(
                True if args.d_dataset != "voc07" else False
            ),  # For VOC07 we only evaluate validation mAP during linear warmup.
        )

        model_d = lw_trainer.conduct_linear_warmup(args)
        del lw_trainer

        model_d = model_d.module  # Unwrap DDP

    if bool(args.use_grad_checkpointing):
        model_d.backbone.set_grad_checkpointing(True)  # type: ignore
        model_p.backbone.set_grad_checkpointing(True)  # type: ignore

    model_d = model_d.to(device)
    model_p = model_p.to(device)

    # Using a frozen common foundation model with LoRA adapters, we can tie the backbone weights of the pretext
    # and downstream model to reduce memory usage, since the backbone weights are not updated during training.
    # This allows us to only store one copy of the backbone weights in memory. We have to do this after sending
    # the models to the device, but after wrapping in DDP, to ensure the weights are tied on each device correctly.
    if args.lora_rank is not None:
        not_tie_keys = ["lora"]
        if bool(args.unfreeze_norm_layers):
            not_tie_keys += ["norm"]
        tie_models(model_p.backbone, model_d.backbone, not_tie_keys)  # type: ignore

    ## DDP WRAP ##
    model_p = wrap_model_for_ddp(model_p)
    model_d = wrap_model_for_ddp(model_d)  # type: ignore

    ### TRAINING SETUP  ###

    # Fetching pretext dataloader and sampler
    pretext_data_transform = MAEPretextTransform(
        img_size=args.p_img_size or pretrain_config.img_size,
        img_crop_min_ratio=args.p_img_crop_min_ratio
        or pretrain_config.img_crop_min_ratio,
    )

    p_dataset_name: PretrainDatasets = args.p_dataset or pretrain_config.dataset
    dataloader_p, sampler_p = get_dataloader_and_sampler(
        args=args,
        dataset_name=p_dataset_name,
        split="train" if "imagenet" in p_dataset_name else "unlabeled",
        batch_size=args.p_batch_size or pretrain_config.batch_size,
        transform=pretext_data_transform,
        drop_last=True,
        img_size=args.p_img_size or pretrain_config.img_size,
    )

    # Fetching optimizers and lr schedulers

    optimizer_p, lr_scheduler_p = get_lower_pretext_optimizer_and_lr_sched(
        args=args,
        pretrain_config=pretrain_config,
        parameter_groups=add_weight_decay(
            model=model_p.backbone,  # type: ignore
            weight_decay=args.p_wd if args.p_wd is not None else pretrain_config.wd,
            lr=args.p_lr,
        )
        + add_weight_decay(
            model=model_p.head,  # type: ignore
            weight_decay=args.p_wd if args.p_wd is not None else pretrain_config.wd,
            lr=args.p_head_lr if args.p_head_lr is not None else args.p_lr,
        ),
    )
    optimizer_d, lr_scheduler_d = get_upper_downstream_optimizer_and_lr_sched(
        args=args,
        parameter_groups=param_groups_lrd(  # type: ignore
            model_d.backbone,  # type: ignore
            weight_decay=args.d_wd,
            no_weight_decay_list=list(model_d.backbone.no_weight_decay()),  # type: ignore
            layer_decay=(  # Do not apply layer decay when using LoRA, as only the adapter layers are being trained.
                1.0 if args.lora_rank is not None else 0.75
            ),
        )
        + [
            dict(
                params=model_d.head.parameters(),  # type: ignore
                weight_decay=args.d_wd,
            )
        ],
    )

    bissl_trainer = BiSSLightClassification(
        optimizers={"upper": optimizer_d, "lower": optimizer_p},
        models={"upper": model_d, "lower": model_p},
        dataloaders={"upper": dataloader_d_train, "lower": dataloader_p},
        device=device,
        ij_grad_calc=IJGradCalcMFACLW(
            parameters=tuple(
                param for param in model_p.backbone.parameters() if param.requires_grad  # type: ignore
            ),
            param_groups_idx=get_lora_layerwise_param_group_idxs(
                model_d.backbone, grouping=args.ij_solver_lora_param_grouping  # type: ignore
            ),
        ),
        logger=logger,
        input_processors={
            "upper": get_upper_input_processor(device),
            "lower": get_lower_input_processor_single(device),
        },
        samplers={"upper": sampler_d_train, "lower": sampler_p},
        lr_schedulers={"upper": lr_scheduler_d, "lower": lr_scheduler_p},
        num_iters={"upper": args.upper_num_iter, "lower": args.lower_num_iter},
        use_amp=args.use_amp,
        ext_bb_mae=True,
        ij_grad_calc_device=(
            device if args.ij_grad_calc_device == "cuda" else torch.device("cpu")
        ),
    )

    print("")
    print("BiSSL Training")

    best_top1_acc = 0
    best_top5_acc = 0
    best_top1_acc_pbb = 0
    best_top5_acc_pbb = 0
    cuda_memory_peak_max = 0

    for epoch in range(args.epochs):
        print("")
        print(f"Epoch {epoch+1} / {args.epochs}\n-------------------------------")

        torch.cuda.reset_peak_memory_stats()

        bissl_trainer.train_one_epoch(args.lam)

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

        dist.all_reduce(cuda_memory_peak, op=dist.ReduceOp.SUM)

        if args.rank == 0:
            cuda_memory_peak_max = max(cuda_memory_peak_max, cuda_memory_peak.item())

        print("")

        if (epoch + 1) % args.eval_interval == 0:
            top1_dbb, top5_dbb, loss_dbb = eval_fn(
                args=args,
                model=model_d,
                dataloader=dataloader_d_val,
                device=device,
                loss_fn=bissl_trainer.loss_fn,
                use_amp=args.use_amp,
            )

            # We also document downstream validation accuracy using the lower-level backbone instead:
            model_d_pbb = deepcopy(model_d.module)
            model_d_pbb.backbone.load_state_dict(model_p.module.backbone.state_dict())  # type: ignore
            model_d_pbb.to(device)
            if args.lora_rank is not None:
                tie_models(model_p.backbone, model_d_pbb.backbone, not_tie_keys)  # type: ignore
            model_d_pbb = wrap_model_for_ddp(model_d_pbb, device)  # type: ignore
            top1_pbb, top5_pbb, loss_pbb = eval_fn(
                args=args,
                model=model_d_pbb,
                dataloader=dataloader_d_val,
                device=device,
                loss_fn=bissl_trainer.loss_fn,
                use_amp=args.use_amp,
            )
            del model_d_pbb
            torch.cuda.empty_cache()

            best_top1_acc = max(best_top1_acc, top1_dbb)
            best_top5_acc = max(best_top5_acc, top5_dbb)
            best_top1_acc_pbb = max(best_top1_acc_pbb, top1_pbb)
            best_top5_acc_pbb = max(best_top5_acc_pbb, top5_pbb)

            log_data_and_state_dicts(
                args=args,
                logger=logger,
                model_p=model_p,
                storage_paths_dict=storage_paths_dict,
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
                cuda_memory_peak=cuda_memory_peak_max,
            )

            cuda_memory_peak_max = 0

    logger.close()
