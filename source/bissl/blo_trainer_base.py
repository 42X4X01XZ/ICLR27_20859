from typing import Any, Callable, Iterator, Literal, Optional, Tuple, Dict, get_args
from abc import abstractmethod

import numpy as np
import torch
import torch.distributed
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import _BaseDataLoaderIter

from source.trainers import CosineLRSchedulerWithWarmup
from source.types import UpperLower
from source.bissl.ij_solver_cg import IJGradCalc
from source.logging import BaseLogger


class GeneralBLOTrainer:
    """
    A general trainer class for bilevel optimization (BLO) tasks. This primarily acts
    as a template for implementing trainers that use BLO, such as BiSSL.

    Attributes:
        optimizers (Dict[UpperLower, torch.optim.Optimizer]): A dictionary containing the optimizers for the upper and lower levels.
        models (Dict[UpperLower, torch.nn.Module]): A dictionary containing the models for the upper and lower levels.
        dataloaders (Dict[UpperLower, DataLoader]): A dictionary containing the dataloaders for the upper and lower levels.
        device (torch.device): The device (CPU or GPU) on which the models and data will be loaded.
        ij_grad_calc (IJGradCalc): An instance of IJGradCalc used for gradient calculations.
        verbose (bool): If True, enables verbose logging.
        logger (Optional[BaseLogger]): A logger instance for training metrics (e.g., WandBLogger or ConsoleLogger).
        input_processors (Dict[UpperLower, Optional[Callable]]): A dictionary containing optional input processors for the upper and lower levels.
            The input processors operate on data points as they are loaded from the dataloader, and outputs the data in a format and on the correct device
            that can be used by the model. They should hence only be None if the data (as the come out of the dataloader) is already in the correct format
            and on the correct device, as that in that instance will be passed directly to the model.
        samplers (Dict[UpperLower, Optional[DistributedSampler]]): A dictionary containing optional samplers for the upper and lower levels.
        lr_schedulers (Dict[UpperLower, Optional[CosineLRSchedulerWithWarmup]]): A dictionary containing optional learning rate schedulers for the upper and lower levels.
        num_iters (Dict[UpperLower, int]): A dictionary specifying the number of iterations for the upper and lower levels.

    """

    def __init__(
        self,
        optimizers: Dict[UpperLower, torch.optim.Optimizer],
        models: Dict[UpperLower, torch.nn.parallel.DistributedDataParallel],
        dataloaders: Dict[UpperLower, DataLoader],
        device: torch.device,
        ij_grad_calc: IJGradCalc,
        logger: Optional[BaseLogger] = None,
        input_processors: Dict[UpperLower, Optional[Callable]] = {
            "upper": None,
            "lower": None,
        },
        samplers: Dict[UpperLower, Optional[DistributedSampler]] = {
            "upper": None,
            "lower": None,
        },
        lr_schedulers: Dict[UpperLower, Optional[CosineLRSchedulerWithWarmup]] = {
            "upper": None,
            "lower": None,
        },
        num_iters: Dict[UpperLower, int] = {"upper": 8, "lower": 20},
    ):
        # Ensure that the input dictionaries have the correct keys
        assert all(
            input_dict.keys() == {"upper", "lower"}
            for input_dict in [
                optimizers,
                models,
                dataloaders,
                input_processors,
                samplers,
                lr_schedulers,
            ]
        ), (
            "The input dictionaries 'optimizers', 'models', 'dataloaders', 'input_processors', 'samplers' "
            + "and 'lr_schedulers' must (exactly) contain the keys 'upper' and 'lower'."
        )
        assert all(
            isinstance(val, torch.optim.Optimizer) for val in optimizers.values()
        ), "Wrong type for 'optimizers'."
        assert all(
            isinstance(val, torch.nn.Module) for val in models.values()
        ), "Wrong type for 'models'."
        assert all(
            isinstance(val, DataLoader) for val in dataloaders.values()
        ), "Wrong type for 'dataloaders'."
        assert all(
            isinstance(val, (DistributedSampler, type(None)))
            for val in samplers.values()
        ), "Wrong type for 'samplers'."
        assert all(
            isinstance(val, (CosineLRSchedulerWithWarmup, type(None)))
            for val in lr_schedulers.values()
        ), "Wrong type for 'lr_schedulers'."

        # Ensuring the models have the required attributes, i.e. being able to access the backbone and head modules
        assert hasattr(models["lower"], "backbone") and isinstance(
            models["lower"].backbone, torch.nn.Module
        ), "The lower model must have a 'backbone' attribute of type torch.nn.Module."
        assert hasattr(models["upper"], "backbone") and isinstance(
            models["upper"].backbone, torch.nn.Module
        ), "The upper model must have a 'backbone' attribute of type torch.nn.Module."
        assert hasattr(models["upper"], "head") and isinstance(
            models["upper"].head, torch.nn.Module
        ), "The upper model must have a 'head' attribute of type torch.nn.Module."

        assert len(dataloaders["lower"]) > 0, "The lower dataloader is empty."
        assert len(dataloaders["upper"]) > 0, "The upper dataloader is empty."

        # assert all(
        #     iter_num > 0 for iter_num in num_iters.values()
        # ), "The number of lower and upper level iterations must be respectively be greater than 0."

        self.optimizers = optimizers
        self.models = models
        self.dataloaders = dataloaders
        self.input_processors = input_processors
        self.samplers = samplers

        self.device = device

        self.ij_grad_calc = ij_grad_calc

        self.lr_schedulers = lr_schedulers

        self.num_iters = num_iters

        self.logger = logger

    def _accumulate_tensor(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Accumulates a tensor across all processes in distributed.
        """
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            tensor = tensor.contiguous()
            # Ensure all tensors in vecs are contiguous
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            return tensor.div_(torch.distributed.get_world_size())
        return tensor

    def _update_grad(
        self,
        grads: Tuple[torch.Tensor, ...] | Iterator[torch.Tensor],
        params: Iterator[torch.Tensor],
    ):
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            params = tuple(params)  # type: ignore
        # Step 1: Accumulate gradients
        for param, grad in zip(params, grads):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.add_(grad)

        # Step 2: Synchronize gradients across gpus
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.cuda.synchronize()
            for param in params:
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad, op=torch.distributed.ReduceOp.SUM
                    )
                    param.grad.div_(torch.distributed.get_world_size()).detach_()

    @abstractmethod
    def _upper_train_loop(self, *args, **kwargs):
        pass

    @abstractmethod
    def _lower_train_loop(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_one_epoch(self, *args, **kwargs):
        pass


class GeneralBiSSLTrainer(GeneralBLOTrainer):
    """A general base class for BiSSL trainers to inherit from."""

    def __init__(self, epoch: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = epoch or 0

        # Init samplers if defined
        for training_stage in get_args(UpperLower):
            if self.samplers[training_stage] is not None:
                self.samplers[training_stage].set_epoch(self.epoch)  # type: ignore

        # Number of times to iterate trough the respective dataloaders before alternating to the other level
        self.dataloader_epochs: Dict[UpperLower, int] = {
            training_stage: int(
                np.ceil(
                    self.num_iters[training_stage]
                    / len(self.dataloaders[training_stage])
                )
            )
            for training_stage in get_args(UpperLower)
        }

        # Contains the iterators for the respective dataloaders. If the number of epochs is 1, the iterator is None, as we will use the dataloader directly then.
        self.dataloader_iterators: Dict[UpperLower, Optional[_BaseDataLoaderIter]] = {
            training_stage: (
                iter(self.dataloaders[training_stage])
                if self.dataloader_epochs[training_stage] == 1
                else None
            )
            for training_stage in get_args(UpperLower)
        }

        # Placeholder for the current step in the dataloader.
        self.dataloader_step: Dict[UpperLower, int] = (
            {  # Placeholder for the current step in the dataloader.
                "lower": 0,
                "upper": 0,
            }
        )

        # Placeholder for the average loss over the number of iterations per training stage.
        # Are reset at the beginning of each training stage.
        # Lower
        self.loss_p_avg = 0.0
        self.loss_reg_avg = 0.0
        # Upper
        self.loss_d_ij_avg = 0.0
        self.loss_d_classic_avg = 0.0

    # ADAPT THIS TO THE SPECIFIC DOWNSTREAM TASK
    @abstractmethod
    def upper_criterion_ext_backbone(
        self,
        upper_input: Tuple[Any, ...],
        ext_backbone: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """This function takes the upper-level input and outputs the differentiable loss for the upper
        level model. Importantly, the function should also be able to exchange the backbone model with an
        external, architecturally identical backbone
        (i.e. isinstance(self.models["upper"].backbone, type(ext_backbone)) should be True). The function
        should be adapted to the specific downstream task.
        """
        # Adapt this function specifically for the downstream task model
        # Should be able to insert an external backbone model (the ext_backbone argument)

        pass

    def _regularization_loss(
        self,
        model_pars: Tuple[torch.nn.Parameter, ...] | Iterator[torch.nn.Parameter],
        w_0: Tuple[torch.nn.Parameter, ...] | Iterator[torch.nn.Parameter],
        lam: float = 1.0,
        which_arg_detach: Literal["first", "second", "none"] = "second",
    ) -> torch.Tensor:

        assert (
            isinstance(lam, float) and lam > 0.0
        ), "The regularization parameter 'lam' must be a positive float/Tensor."

        match which_arg_detach:
            case "first":
                deltas = tuple(
                    param.clone().detach() - w_0_par
                    for param, w_0_par in zip(model_pars, w_0)
                )
            case "second":
                deltas = tuple(
                    param - w_0_par.clone().detach()
                    for param, w_0_par in zip(model_pars, w_0)
                )
            case "none":
                deltas = tuple(
                    param - w_0_par for param, w_0_par in zip(model_pars, w_0)
                )

        return 0.5 * lam * sum(torch.sum(delta**2) for delta in deltas)  # type: ignore

    def upper_log(self, loss_ig: float, loss_classic: float, epoch: int):
        print(
            f"Avg loss over batch no. "
            + f' {self.dataloader_step["upper"] + 1 - self.num_iters["upper"]}-{self.dataloader_step["upper"]}'
            + f' / {len(self.dataloaders["upper"])}: {loss_ig + loss_classic:>5f}'
        )
        if self.logger:
            grad_abs_medians = {
                name: torch.median(par.grad.clone().detach().abs()).item()
                for name, par in self.models["upper"].named_parameters()
                if par.grad is not None
            }
            grad_norms = {
                name: torch.norm(par.grad.clone().detach(), p=2).item()
                for name, par in self.models["upper"].named_parameters()
                if par.grad is not None
            }
            self.logger.log(
                {
                    "train/upper_loss_ij_train": loss_ig,
                    "train/upper_loss_classic_train": loss_classic,
                    "train/upper_loss_total_train": loss_ig + loss_classic,
                    "train/upper_lr": self.optimizers["upper"].param_groups[0]["lr"],
                    "train/upper_grad_abs_median": grad_abs_medians,
                    "train/upper_grad_norms": grad_norms,
                    "train/epoch": epoch,
                }
            )

    def lower_log(
        self,
        loss_ssl: float,
        loss_reg: float,
        lam: float,
        epoch: int,
    ):
        print(
            "Avg loss over batch no. "
            + f'{self.dataloader_step["lower"] + 1 - self.num_iters["lower"]}-{self.dataloader_step["lower"]}'
            + f' / {len(self.dataloaders["lower"])}: {loss_ssl:>5f}'
        )

        assert (
            isinstance(lam, float) and lam > 0.0
        ), "The regularization parameter 'lam' must be a positive float/Tensor."

        if self.logger:
            grad_abs_medians = {
                name: par.grad.abs().median().item()
                for name, par in self.models["lower"].named_parameters()
                if par.grad is not None
            }
            grad_norms = {
                name: par.grad.norm(p=2).item()
                for name, par in self.models["lower"].named_parameters()
                if par.grad is not None
            }
            log_dict = {
                "train/lower_loss_train": loss_ssl,
                "train/lower_reg_loss_train": loss_reg,
                "train/lower_total_loss_train": loss_ssl + lam * loss_reg,
                "train/lower_lr": self.optimizers["lower"].param_groups[0]["lr"],
                "train/lower_grad_abs_median": grad_abs_medians,
                "train/lower_grad_norms": grad_norms,
                "train/lam": lam,
                "train/epoch": epoch,
            }

            self.logger.log(log_dict)

    def _calibrate_dataloader(self, training_level: UpperLower):
        """Calibrates the dataloader for the respective training level, ensuring that it contains
        sufficient batches for succesfully executing the number of iterations per training stage.
        """
        # If the number of epochs for the dataloader is greater than 1, we simply use the dataloader directly, as seen in "train_one_epoch".
        if self.dataloader_epochs[training_level] > 1:
            self.dataloader_step[training_level] = 0

        # Otherwise, we store the dataloader iterator, and only resets it in case the number of iterations is less than the number of iterations per training stage iteration.
        # (E.g. if we have 5 iterations per a training stage, and the dataloader iterator only has 3 batches left, we reset it)
        elif (
            len(self.dataloaders[training_level]) - self.dataloader_step[training_level]
            < self.num_iters[training_level]
        ):
            if self.samplers[training_level] is not None:
                self.samplers[training_level].set_epoch(self.epoch)  # type: ignore

            self.dataloader_iterators[training_level] = iter(
                self.dataloaders[training_level]
            )
            self.dataloader_step[training_level] = 0

    def _train_one_level(
        self, training_stage: UpperLower, trainer_fn: Callable, **kwargs
    ):
        for train_iter in range(self.dataloader_epochs[training_stage]):
            if (
                self.samplers[training_stage] is not None
                and self.dataloader_epochs[training_stage] > 1
            ):
                self.samplers[training_stage].set_epoch(  # type: ignore
                    self.epoch * self.dataloader_epochs[training_stage] + train_iter
                )
            trainer_fn(
                dataloader=(  # ignore type here as we know they are not None # type: ignore
                    self.dataloader_iterators[training_stage]
                    if self.dataloader_iterators[training_stage] is not None
                    else self.dataloaders[training_stage]
                ),
                epoch=self.epoch,
                **kwargs,
            )
