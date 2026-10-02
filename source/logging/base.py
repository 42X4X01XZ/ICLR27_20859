from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLogger(ABC):
    """Abstract base class for logging training metrics."""

    @abstractmethod
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics (e.g., loss, accuracy) at a given step."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources (e.g., close WandB run)."""
        pass
