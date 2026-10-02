import os
from typing import Any, Callable, Dict, Optional

from itertools import chain
import random

import torch

from ray import tune, init
from ray.tune import TuneConfig
from ray.tune.search.basic_variant import BasicVariantGenerator

from runs.fine_tune.config import ArgsFineTuningDefaults


class RandomizedVariantGenerator(BasicVariantGenerator):
    """Randomizes the order of grid search trials in BasicVariantGenerator."""

    def add_configurations(self, experiments):
        # Build iterators as usual
        super().add_configurations(experiments)

        # Flatten all _TrialIterator objects into a list of individual trial iterators
        trial_lists = [list(it) for it in self._iterators]
        # Shuffle all trials
        shuffled_trials = list(chain.from_iterable(trial_lists))
        random.shuffle(shuffled_trials)

        # Replace the internal trial generator with shuffled trials
        self._trial_generator = iter(shuffled_trials)


def start_hpo(
    args: ArgsFineTuningDefaults,
    num_runs: int,
    train_fn: Callable[[Dict[str, Any]], None],
    hpo_config: Optional[Dict[str, Any]] = None,
    tuner_config: Optional[TuneConfig] = None,
):

    # Measures to prevent ray tune from logging too much data.
    # However, it is to my knowledge impossible to fully disable the logging of the trial results.
    # Hence, make sure to occasionally clean the ray_spill directory manually...
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(10 * torch.cuda.device_count())

    if args.rank == 0:
        # HPO Setup
        if bool(args.use_hpo):
            assert (
                hpo_config is not None
            ), "HPO config must be provided when use_hpo is True."
        else:
            hpo_config = {"lr": tune.choice([args.lr])}

        assert all(
            key in args._get_argument_names() for key in hpo_config.keys()
        ), "Some hpo-configurable hyperparameters are not found in the config."

        ngpus_pr_task = 1  # Ray only supports 1 GPU per task.
        args.world_size = (
            ngpus_pr_task  # Will be the world size for each process spawned by ray.
        )
        init(
            num_cpus=args.num_workers * torch.cuda.device_count(),
            num_gpus=torch.cuda.device_count(),
            include_dashboard=False,
            log_to_driver=False,
        )
        trainable_with_resources = tune.with_resources(
            trainable=train_fn,
            resources={
                "cpu": args.num_workers,
                "gpu": ngpus_pr_task,
            },
        )

        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=hpo_config,
            tune_config=(
                TuneConfig(num_samples=num_runs)
                if tuner_config is None
                else tuner_config
            ),
        )

        tuner.fit()
