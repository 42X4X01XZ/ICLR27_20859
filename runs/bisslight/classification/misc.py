import torch


def tie_models(
    model_base: torch.nn.Module,
    model_to_tie: torch.nn.Module,
    not_tie_keys: list = ["lora"],
) -> None:
    for (name_base, param_base), (name_tie, param_tie) in zip(
        model_base.named_parameters(),  # type: ignore
        model_to_tie.named_parameters(),  # type: ignore
    ):
        if not any(key in name_base for key in not_tie_keys):
            assert (
                name_base == name_tie
            ), "Parameter names do not match between pretext and downstream model"
            param_tie.data = param_base.data
