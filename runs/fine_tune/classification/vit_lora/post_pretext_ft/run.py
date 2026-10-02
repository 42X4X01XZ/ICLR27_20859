from functools import partial

from ray import tune

from runs.distributed import init_dist
from runs.fine_tune.hpo_ray import start_hpo, RandomizedVariantGenerator
from runs.fine_tune.misc import get_pretrain_config_dict
from runs.fine_tune.classification.vit_lora.post_pretext_ft.config import (
    ArgsFTClassificationViTLoraPostPretext,
)
from runs.fine_tune.classification.vit_lora.train import train_classification_vit
from runs.pretext.mae.config import ArgsPretextMAEOfficial

if __name__ == "__main__":
    args = ArgsFTClassificationViTLoraPostPretext().parse_args()

    # Distributed Setup
    _ = init_dist(args, init_process_group=False)

    ### Fetch Pretrain Config and Init Wandb ###
    pretext_config = ArgsPretextMAEOfficial()

    ### Fetch the pretrained backbone model architecture ###
    match args.pretrained_backbone_path:
        case s if s.endswith("mae_pretrain_vit_base_full.pth"):
            backbone_arch = "vit_base_patch16"
        case s if s.endswith("mae_pretrain_vit_large_full.pth"):
            backbone_arch = "vit_large_patch16"
        case s if s.endswith("mae_pretrain_vit_huge_full.pth"):
            backbone_arch = "vit_huge_patch14"
        case _:
            raise ValueError(
                f"Unsupported pretrained_backbone_path: {args.pretrained_backbone_path}"
                + "Dir has to end with (one of the official mae pretrained checkpoints): 'mae_pretrain_vit_base_full.pth', 'mae_pretrain_vit_large_full.pth', 'mae_pretrain_vit_huge_full.pth'"
            )
    # Official values read off from https://arxiv.org/abs/2111.06377
    pretext_config.from_dict(
        {"dataset_root": args.dataset_root, "backbone_arch": backbone_arch}
    )

    # The main trainer script.
    trainer_fn = partial(
        train_classification_vit,
        args=args,
        backbone_arch=pretext_config.backbone_arch,  # Already asserted the backbone is compatible # type: ignore
        pretrained_bb_path=args.pretrained_backbone_path,
        config_dict_to_upload=get_pretrain_config_dict(pretext_config),
        seed=args.seed,
    )

    if args.use_hpo:
        assert (
            args.lr is None
        ), "Alr has to be None when use_hpo is True, since its being tuned. Alternatively, set use_hpo to False if you want to specify lr manually."

        lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

        hpo_config = {"lr": tune.grid_search(lrs)}
        tuner_config = tune.TuneConfig(search_alg=RandomizedVariantGenerator())
        args.num_runs = len(lrs)
    else:
        assert args.lr is not None, "lr has to be specified when hpo is not executed."
        hpo_config = None
        tuner_config = None

    # Start the HPO process, using the trainer function.
    start_hpo(
        args=args,
        train_fn=trainer_fn,
        hpo_config=hpo_config,
        tuner_config=tuner_config,
        num_runs=args.num_runs,
    )
