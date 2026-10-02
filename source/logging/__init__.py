from .base import BaseLogger
from .console import ConsoleLogger
from .wandb import WandBLogger

def get_logger(logger_type: str, **kwargs) -> 'BaseLogger':
    """Factory function to create a logger based on type."""
    if logger_type == "wandb":
        return WandBLogger(**kwargs)
    elif logger_type == "console":
        return ConsoleLogger(**kwargs)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}. Use 'wandb' or 'console'.")