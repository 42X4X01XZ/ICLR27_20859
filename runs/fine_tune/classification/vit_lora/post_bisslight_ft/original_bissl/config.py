from typing import Literal
from runs.fine_tune.classification.vit_lora.post_bisslight_ft.config import (
    ArgsFTClassificationViTLoraPostBiSSLight,
)


class ArgsFTClassificationViTLoraPostOrigBiSSL(
    ArgsFTClassificationViTLoraPostBiSSLight
):
    # fmt: off
    #### ADJUSTED DEFATULS ####
    run_type: Literal["ft-post-bissl"] = "ft-post-bissl" # type: ignore

    # fmt: on
