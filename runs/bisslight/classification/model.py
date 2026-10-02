from typing import Callable, Tuple
import torch


def get_upper_input_processor(
    device: torch.device,
) -> Callable:
    def return_fn(upper_input: Tuple[torch.Tensor, torch.Tensor]):
        imgs, labels = upper_input
        return imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    return return_fn
