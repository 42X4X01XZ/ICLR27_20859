# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import numpy as np
import torch
from runs.config_general import ArgsGeneralDefaults, SEED_MAX_VALUE


def generate_seed(args: ArgsGeneralDefaults) -> int:
    """
    Generate a seed on main process and broadcasts it to all remaining processes.
    """
    if args.world_size > 1:
        seed_list = [None]
        if args.rank == 0:
            seed_list = [random.randint(0, SEED_MAX_VALUE)]
        torch.distributed.broadcast_object_list(object_list=seed_list, src=0)
        torch.distributed.barrier()
        assert seed_list[0] is not None
        return seed_list[0]

    return random.randint(0, SEED_MAX_VALUE)


def set_seed(args: ArgsGeneralDefaults, force_generate_new_seed: bool = False) -> None:
    if args.seed is None or force_generate_new_seed:
        args.seed = generate_seed(args)

    print(f"Using seed: {args.seed}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_dist(args, init_process_group=True):
    if bool(args.cudnn_benchmark):
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Setting ENV variables
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    # Achieving distributed parameters
    if (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and "LOCAL_RANK" in os.environ
    ):
        assert (
            os.environ["WORLD_SIZE"] == os.environ["LOCAL_WORLD_SIZE"]
        ), "Not implemented for multi-node training."

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        raise ValueError("Not using distributed mode")

    if init_process_group:
        # Starting dist process
        print(
            "| distributed init (rank {}): {}, gpu {}, world size {}".format(
                args.rank,
                args.dist_url,
                args.gpu,
                args.world_size,
            ),
            flush=True,
        )
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
        )
        torch.cuda.set_device(args.gpu)
        torch.distributed.barrier()

        set_seed(args)

    # Enables printing only for rank 0
    setup_for_distributed(args.rank == 0)

    # Returns device
    return torch.device(args.gpu)
