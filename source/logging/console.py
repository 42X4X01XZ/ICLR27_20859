import logging
from typing import Dict, Any, Optional
from .base import BaseLogger


class ConsoleLogger(BaseLogger):
    """Logs metrics to the console."""

    def __init__(
        self,
        level: int = logging.INFO,
        config: Optional[Dict[str, Any]] = None,
        rank: Optional[int] = None,
    ):
        self.rank = rank
        self.config = config  # Store config
        self.logger = logging.getLogger("ConsoleLogger")
        self.logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to console, only on rank 0 if a rank is given."""
        if self.rank is not None and self.rank != 0:
            return
        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            elif isinstance(v, dict):
                # For dictionaries, just log the key name and count to avoid verbose output
                parts.append(f"{k}=dict({len(v)} items)")
            else:
                parts.append(f"{k}={v}")
        message = f"Step {step}: " + ", ".join(parts)
        self.logger.info(message)

    def close(self) -> None:
        """No cleanup needed for console logging."""
        pass
