from typing import Literal
from runs.fine_tune.classification.vit_lora.config import ArgsFTClassificationViTLora


class ArgsFTClassificationViTLoraPostPretext(ArgsFTClassificationViTLora):
    # fmt: off
    #### ADJUSTED DEFATULS ####
    run_type: Literal["ft-post-pretext"] = "ft-post-pretext" # type: ignore

    #### PostPretext SPECIFIC ARGUMENTS ####

    # fmt: on
