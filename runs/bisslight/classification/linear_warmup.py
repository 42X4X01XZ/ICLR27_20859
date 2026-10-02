from source.trainers import train_classifier
from source.evals import (
    eval_classifier_top1_and_top5,
    eval_classifier_mAP,
)

from runs.bisslight.classification.config import ArgsBiSSLClassification
from runs.bisslight.linear_warmup import LinearWarmupTrainer


class ClassifierLinearWarmupTrainer(LinearWarmupTrainer):
    def __init__(
        self, use_amp: bool = False, eval_accuracy: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.eval_accuracy = eval_accuracy

    def train_one_epoch(self):
        return train_classifier(
            model=self.model,
            loss_fn=self.loss_fn,
            dataloader=self.dataloader_train,
            optimizer=self.optimizer,
            device=self.device,
            use_amp=self.use_amp,
            eval_accuracy=self.eval_accuracy,
            return_loss_avg=True,
        )

    def eval_fn(self, args: ArgsBiSSLClassification):  # type: ignore
        if self.dataloader_val is None:
            return 0, 0, 0
        elif args.d_dataset == "voc07":
            map, loss = eval_classifier_mAP(
                model=self.model,
                dataloader=self.dataloader_val,
                device=self.device,
                label="Validation Performance (LW, mAP)",
            )
            return map, 0, loss
        return eval_classifier_top1_and_top5(
            model=self.model,
            dataloader=self.dataloader_val,
            loss_fn=self.loss_fn,
            device=self.device,
            label="Validation Performance (LW)",
            use_amp=self.use_amp,
        )
