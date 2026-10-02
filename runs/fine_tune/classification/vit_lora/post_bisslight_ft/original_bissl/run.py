from functools import partial

import yaml
from ray import tune

from runs.bisslight.classification.mae_lora.original_bissl.config import (
    ArgsBiSSLClassificationMAELoraOrigBiSSL,
)
from runs.config_general import ArgsGeneralDefaults
from runs.distributed import init_dist
from runs.fine_tune.classification.vit_lora.post_bisslight_ft.original_bissl.config import (
    ArgsFTClassificationViTLoraPostOrigBiSSL,
)
from runs.fine_tune.hpo_ray import start_hpo, RandomizedVariantGenerator
from runs.fine_tune.misc import (
    get_pretrain_config_dict,
    update_args_postbissl,
)
from runs.fine_tune.classification.vit_lora.train import train_classification_vit
from runs.misc import override_config
from runs.pretext.mae.config import ArgsPretextMAEOfficial

# NOTE: This script is almost identical with runs/fine_tune/classification/vit_lora/post_bisslight_ft/run.py,
# except that it uses the ArgsBiSSLClassificationMAELoraOrigBiSSL and ArgsFTClassificationViTLoraPostOrigBiSSL
# classes instead of the ArgsBiSSLightClassificationMAELora and ArgsFTClassificationViTLoraPostBiSSLight classes.
# This is because the original BiSSL config has different default values for some hyperparameters than the
# BiSSLight config. The rest of the script is identical. While a function could be used to avoid code duplication,
# it would make the code less readable and more difficult to understand. Therefore, we have chosen to keep the
# two scripts separate, even though they are almost identical.

if __name__ == "__main__":
    args = ArgsFTClassificationViTLoraPostOrigBiSSL().parse_args()

    # Distributed Setup
    _ = init_dist(args, init_process_group=False)

    ### BISSL CONFIG LOAD ###
    assert (
        args.run_type == "ft-post-bissl"
    ), "run_type has to be 'ft-post-bissl' for this script."

    assert (
        args.bissl_pretrained_model_path is not None
    ), "bissl_pretrained_model_path has to be specified. This is the path to the pretrained BiSSL model that will be used for fine-tuning."

    pretext_config = ArgsPretextMAEOfficial()

    # Fetches the BiSSLight-specific config.
    bissl_config = ArgsBiSSLClassificationMAELoraOrigBiSSL()

    if args.bissl_config_path is None:
        print(
            "Warning: No BiSSight/BiSSL config path specified. Using default config values. This may lead to incompatability issues if critical values from the BiSSL config used are differ from the ones specified here (e.g. the lora rank, image size, dataset name, etc.)."
        )
    else:
        with open(args.bissl_config_path, "r") as f:
            bissl_config_dict = yaml.safe_load(f)

        override_config(
            bissl_config,
            bissl_config_dict,
            missing_keys_skip=set(list(ArgsGeneralDefaults()._get_argument_names())),
        )

    pretrained_backbone_filename = args.pretrained_backbone_path.split("/")[-1]

    assert (
        pretrained_backbone_filename
        == bissl_config.pretrained_backbone_path.split("/")[-1]
    ), f"The model specified by the pretrained_backbone_path from the BiSSL config ({pretrained_backbone_filename}) does not match the one specified in the args ({args.pretrained_backbone_path.split('/')[-1]}). Please ensure they are consistent."

    ### Fetch the pretrained backbone model architecture ###
    match pretrained_backbone_filename:
        case "mae_pretrain_vit_base_full.pth":
            backbone_arch = "vit_base_patch16"
        case "mae_pretrain_vit_large_full.pth":
            backbone_arch = "vit_large_patch16"
        case "mae_pretrain_vit_huge_full.pth":
            backbone_arch = "vit_huge_patch14"
        case _:
            raise ValueError(
                f"Unsupported pretrained_backbone_path: {args.pretrained_backbone_path}"
                + "Dir has to reference on of the following files (the official mae pretrained checkpoints): 'mae_pretrain_vit_base_full.pth', 'mae_pretrain_vit_large_full.pth', 'mae_pretrain_vit_huge_full.pth'"
            )
    # Official values read off from https://arxiv.org/abs/2111.06377
    pretext_config.from_dict(
        {"dataset_root": args.dataset_root, "backbone_arch": backbone_arch}
    )

    # Override the parsed args with the BiSSL config for specified post-bissl hyperparameters
    # that are None. Non-None hyperparameters are unchanged.
    # See inside the function for which hyperparameters this concerns specifically.
    update_args_postbissl(args, bissl_config, pretext_config)

    if args.lora_rank is not None:
        bissl_fitted_lora_params_path = args.bissl_pretrained_model_path
        pretrained_bb_path = args.pretrained_backbone_path
    else:
        # Assuming Full FT,
        # Then the pretraind model weights have just been stored as the backbone weights, and there is no separate LoRA adapter weights to load.
        bissl_fitted_lora_params_path = None
        pretrained_bb_path = args.bissl_pretrained_model_path

    print(f"Using pretrained backbone model from path: {pretrained_bb_path}")

    # The main trainer script.
    trainer_fn = partial(
        train_classification_vit,
        args=args,
        backbone_arch=pretext_config.backbone_arch,  # Already asserted the backbone is compatible # type: ignore
        pretrained_bb_path=pretrained_bb_path,
        config_dict_to_upload=get_pretrain_config_dict(pretext_config, bissl_config),
        pretrained_lora_path=bissl_fitted_lora_params_path,
        seed=args.seed,
    )

    if args.use_hpo:
        assert (
            args.lr is None
        ), "Alr has to be None when use_hpo is True, since its being tuned. Alternatively, set use_hpo to False if you want to specify lr manually."

        lrs = lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
        hpo_config = {"lr": tune.grid_search(lrs)}
        tuner_config = tune.TuneConfig(search_alg=RandomizedVariantGenerator())
        args.num_runs = len(lrs)
    else:
        assert args.lr is not None, "lr has to be specified when we dont do hpo"
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
