from typing import Literal

from runs.bisslight.classification.mae_lora.config import (
    ArgsBiSSLightClassificationMAELora,
)
from source.types import BinaryChoices


class ArgsBiSSLClassificationMAELoraOrigBiSSL(ArgsBiSSLightClassificationMAELora):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    run_type: Literal["bissl"] = "bissl" # type: ignore[override]
    hinv_solver: Literal["cg_lw"] = "cg_lw"  # The hessian inverse solver to use for the implicit gradient computation. # type: ignore[override]
   
    # CG Solver Specific Args
    cg_lam_dampening: float = 1.0  # Lambda dampening parameter for the cg solver.
    cg_iter_num: int = 5  # Number of cg iterations per iHVP estimate.
    cg_verbose: BinaryChoices = 0  # Whether to print cg solver verbose output.

    # fmt: on
