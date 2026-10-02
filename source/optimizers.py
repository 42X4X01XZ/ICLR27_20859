import torch
from typing import Tuple

ADAM_BETAS_DEFAULTS = (0.9, 0.999)


def get_optimizer(
    optim,
    params,
    lr: float,
    wd: float = 0.0,
    momentum: float = 0.9,
    betas: Tuple[float, float] = ADAM_BETAS_DEFAULTS,
):
    match optim:
        case "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
        case "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
        case "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
        case _:
            raise ValueError("Invalid optimizer choice")
