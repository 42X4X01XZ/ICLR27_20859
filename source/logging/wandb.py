from typing import Dict, Any, Optional
from .base import BaseLogger

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger(BaseLogger):
    """Logs metrics to Weights & Biases."""

    def __init__(
        self,
        project: str,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        rank: Optional[int] = None,
    ):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "WandB is not installed. Install it with `pip install wandb`."
            )
        self.rank = rank
        self.project = project
        self.config = config or {}
        self.run = None

        if self.rank is not None and self.rank != 0:
            return

        if api_key:
            wandb.login(key=api_key)  # type: ignore
        self.run = wandb.init(project=project, config=self.config)  # type: ignore

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB, only on rank 0 if a rank is given."""
        if self.run is None:
            return
        wandb.log(metrics, step=step)  # type: ignore

    def close(self) -> None:
        """Finish the WandB run."""
        if self.run is None:
            return
        self.run.finish()
