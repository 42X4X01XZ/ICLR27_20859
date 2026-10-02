from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader

from source.models.downstream.classification import DSClassifier
from source.bissl.blo_trainer_base import GeneralBiSSLTrainer
from source.bissl.bisslight_trainer import BiSSLightTrainerMFAC
from source.bissl.bissl_trainer import BiSSLTrainerOriginalViTSetting
from source.evals import (
    eval_classifier_top1_and_top5,
    eval_classifier_mAP,
)
from source.models.pretext.mae import MaskedAutoencoderViTBiSSL

from runs.bisslight.classification.config import ArgsBiSSLClassification


class BiSSLClassificationTrainer(GeneralBiSSLTrainer):
    """
    Trainer for downstream classification tasks using BiSSL.

    This class extends GeneralBiSSLTrainer and is specifically designed for
    downstream classification tasks. It uses a cross-entropy loss function
    and expects the model to be an instance of DSClassifier.
    """

    def __init__(self, ext_bb_mae: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert isinstance(self.models["upper"].module, DSClassifier)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # We need this arg for MAEs specifically, to ensure the backbone does not mask
        # patches during the forward pass when computing the upper loss.
        if ext_bb_mae:
            assert isinstance(self.models["lower"].module, MaskedAutoencoderViTBiSSL)
        self.ext_bb_mae = ext_bb_mae

    def upper_criterion_ext_backbone(
        self,
        upper_input: Tuple[torch.Tensor, torch.Tensor],
        ext_backbone: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:

        if ext_backbone is not None:
            # The solution with the MAE backbone is a bit hacky, and could likely
            # be implemented more elegantly.
            if self.ext_bb_mae:
                return self.loss_fn(
                    self.models["upper"].head(
                        ext_backbone(
                            upper_input[0],
                            mask_ratio=0.0,  # type: ignore
                        )
                    ),
                    upper_input[1],
                )
            return self.loss_fn(
                self.models["upper"].head(ext_backbone(upper_input[0])),  # type: ignore
                upper_input[1],
            )
        return self.loss_fn(self.models["upper"](upper_input[0]), upper_input[1])


class BiSSLightClassification(BiSSLClassificationTrainer, BiSSLightTrainerMFAC):
    """Trainer for downstream classification tasks using BiSSLight with the MFAC solver."""


class BiSSLClassificationOriginalViTSetting(
    BiSSLClassificationTrainer, BiSSLTrainerOriginalViTSetting
):
    """Trainer for downstream classification tasks using the original BiSSL with the canonical conjugate gradient (CG) solver."""


def eval_fn(
    args: ArgsBiSSLClassification,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[torch.nn.Module] = None,
    use_amp: bool = False,
):
    if args.d_dataset == "voc07":
        top1_dbb, loss_dbb = eval_classifier_mAP(
            model,
            dataloader,
            device,
            label="Validation Performance (MAP)",
        )
        top5_dbb = 0
    else:
        top1_dbb, top5_dbb, loss_dbb = eval_classifier_top1_and_top5(
            model,
            dataloader,
            loss_fn,  # type: ignore
            device,
            label="Validation Performance",
            use_amp=use_amp,
        )

    return top1_dbb, top5_dbb, loss_dbb
