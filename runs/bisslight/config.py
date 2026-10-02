from typing import Literal, Optional

from source.types import (
    DownstreamDatasets,
    DownstreamTaskTypes,
    PretrainDatasets,
    OptimizerChoices,
    FloatNonNegative,
    FloatBetween0and1,
    HinvSolverTypes,
)

from runs.config_general import ArgsGeneralDefaults
from runs.fine_tune.config import ArgsFineTuningDefaults


class ArgsBiSSLDefaults(ArgsGeneralDefaults):
    # fmt: off
    run_type: Literal["BiSSLight", "BiSSL"]  # Name of the run type

    pretrained_backbone_path: str  # Path to the pretrained backbone model.

    epochs: int = 500  # Number of training stage alternations (here called epochs) to train the downstream task
    downstream_task: DownstreamTaskTypes  # Type of downstream task.
    
    eval_interval: int = 10 # Evaluate the model every n (eval_n_epoch) epochs. Is 20 by default, but the high computational cost of semantic segmentation eval inspired the implementation to eval at less regular intervals.

    # Downstream Data
    d_dataset: DownstreamDatasets  # Downstream dataset name.
    d_batch_size: int  # Downstream batch size.

    # Downstream Training
    d_lr: FloatNonNegative  # Upper level (downstream) learning rate.
    d_wd: FloatNonNegative  # Upper level (downstream) weight decay.
    d_momentum: FloatBetween0and1 = ArgsFineTuningDefaults.momentum  # Upper level (downstream) momentum.
    d_optimizer: OptimizerChoices = ArgsFineTuningDefaults.optimizer # Upper level (downstream) optimizer.
    d_beta1: FloatBetween0and1 = ArgsFineTuningDefaults.beta1  # Upper level (downstream) beta1 for adam optimizers
    d_beta2: FloatBetween0and1 = ArgsFineTuningDefaults.beta2  # Upper level (downstream) beta2 for adam optimizers

    # Downstream Linear Warmup Hyperparameters
    d_head_warmup_epochs: int = 20  # Number of epochs for the linear downstream head warmup.
    d_head_warmup_lr: Optional[FloatNonNegative] = None  # Downstream linear warmup learning rate. Defaults to the downstream learning rate d_lr.
    d_head_warmup_wd: Optional[FloatNonNegative] = None  # Downstream linear warmup weight decay. Defaults to the downstream weight decay d_wd.
    d_head_warmup_momentum: Optional[FloatBetween0and1] = None  # Downstream linear warmup momentum. Defaults to the downstream momentum d_momentum.

    # Remaining optionals are set equal to the corresponding hyperparameters used for pretraining.
    # Pretext Data
    p_batch_size: Optional[int] = None  # Lower level (pretext) batch size. Defaults to the batch size used for pretraining.
    p_dataset: Optional[PretrainDatasets] = None  # Lower level (pretext) dataset name. Defaults to the dataset used for pretraining.

    # Pretext Training
    p_lr: FloatNonNegative  # Lower level (pretext) learning rate.

    p_wd: Optional[FloatNonNegative] = None  # Lower level (pretext) weight decay. Defaults to the weight decay used for pretraining.
    p_momentum: Optional[FloatBetween0and1] = None  # Lower level (pretext) momentum. Defaults to the momentum used for pretraining.
    p_optimizer: Optional[OptimizerChoices] = None  # Lower level (pretext) optimizer. Defaults to the optimizer used for pretraining.
    p_beta1: Optional[FloatBetween0and1] = None  # Lower level (pretext) beta1 for adam optimizers. Defaults to the beta1 used for pretraining.
    p_beta2: Optional[FloatBetween0and1] = None  # Lower level (pretext) beta2 for adam optimizers. Defaults to the beta2 used for pretraining.
    
    # BiSSL Hyperparameters
    lower_num_iter: int = 20  # Number of lower level (pretext) training iterations.
    upper_num_iter: int = 8  # Number of upper level (downstream) training iterations.
    hinv_solver: HinvSolverTypes  # Hessian inverse solver type.
    lam: float =  0.001  # Lambda parameter for the lower level regularization scaling.

    # fmt: on
