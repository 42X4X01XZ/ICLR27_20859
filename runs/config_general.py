from typing import Optional
from tap import Tap
from source.types import BinaryChoices

SEED_MAX_VALUE = 2**30


# In tap, argument help strings are added as a comment on the same line as its corresponding argument
class ArgsGeneralDefaults(Tap):
    # fmt: off
    dataset_root: str  # Path to dataset root dir.
    model_root: Optional[str] = None  # Optional path to root dir for saving models. If None, none are saved.
    
    wandb_api_key: Optional[str] = None # Wandb API key

    download: BinaryChoices = 0  # Download dataset if not found in root

    # device: Literal["cuda"] = "cuda"  # Device, currently only supports cuda
    num_workers: int = 8  # Number of workers for dataloader

    # Distributed Setup
    omp_num_threads: int = 1  # Number of threads
    dist_backend = "nccl"  # Backend for distributed training
    dist_url: str = "env://"  # URL for distributed training

    rank: int = 0  # AKA global rank, will be overwriten during runtime
    gpu: int = 0  # AKA local rank, will be overwriten during runtime
    world_size: int = 1  # Number of GPUs per node, will be overwriten during runtime

    cudnn_benchmark: BinaryChoices = 0  # Use cudnn benchmark. May lead to improved training speed, but can be of the cost of reproducibility.

    seed: Optional[int] = None  # Optional seed for random number generator. If None, the script will overwrite this value with a randomly generated seed, that is stored in the wandb config.

    # fmt: on

    def __init__(self, *args, **kwargs):
        super().__init__(underscores_to_dashes=True, *args, **kwargs)
