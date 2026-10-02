from typing import Any, Optional
from abc import abstractmethod
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from source.logging.base import BaseLogger
from source.optimizers import get_optimizer

from runs.bisslight.config import ArgsBiSSLDefaults


class LinearWarmupTrainer:
    def __init__(
        self,
        args: ArgsBiSSLDefaults,
        logger: BaseLogger,
        model: torch.nn.Module,
        dataloader_train: DataLoader,
        sampler_train: Optional[DistributedSampler],
        dataloader_val: Optional[DataLoader],
        sampler_val: Optional[DistributedSampler],
        device: torch.device,
        loss_fn: Optional[torch.nn.Module] = None,
        lr_scheduler=None,
    ):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.sampler_train = sampler_train
        self.sampler_val = sampler_val
        self.device = device
        self.lr_scheduler = lr_scheduler

        self.optimizer = get_optimizer(
            args.d_optimizer,
            params=model.head.parameters(),  # type: ignore
            lr=args.d_head_warmup_lr or args.d_lr,
            wd=args.d_head_warmup_wd or args.d_wd,
            momentum=args.d_head_warmup_momentum or args.d_momentum,
            betas=(args.d_beta1, args.d_beta2),
        )

        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        self.logger = logger

    @abstractmethod
    def train_one_epoch(self) -> Any:
        pass

    @abstractmethod
    def eval_fn(self, args: ArgsBiSSLDefaults) -> Any:
        pass

    def conduct_linear_warmup(
        self,
        args: ArgsBiSSLDefaults,
    ) -> torch.nn.Module:

        best_acc_lw = argparse.Namespace(top1=0, top5=0)

        par_req_grad = tuple(p.requires_grad for p in self.model.backbone.parameters())  # type: ignore
        for par in self.model.backbone.parameters():  # type: ignore
            par.requires_grad = False

        print("Linear Warmup Training")
        for w_epoch in range(args.d_head_warmup_epochs):
            print("")
            print(
                f"Warmup Epoch {w_epoch+1} / {args.d_head_warmup_epochs}\n-------------------------------"
            )

            if self.sampler_train is not None:
                self.sampler_train.set_epoch(
                    w_epoch + args.epochs * args.upper_num_iter + 1
                )
            if self.sampler_val is not None:
                self.sampler_val.set_epoch(
                    w_epoch + args.epochs * args.upper_num_iter + 1
                )

            self.train_one_epoch()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if (w_epoch + 1) % max((args.d_head_warmup_epochs // 5), 1) == 0:
                print("")

                top1_lw_tr, top5_lw_tr, loss_lw_tr = self.eval_fn(args)

                best_acc_lw.top1 = max(best_acc_lw.top1, top1_lw_tr)
                best_acc_lw.top5 = max(best_acc_lw.top5, top5_lw_tr)

                self.logger.log(
                    {
                        "lw_test/top1_acc_test": top1_lw_tr,
                        "lw_test/top5_acc_test": top5_lw_tr,
                        "lw_test/best_top1_acc_test": best_acc_lw.top1,
                        "lw_test/best_top5_acc_test": best_acc_lw.top5,
                        "lw_test/avgloss_test": loss_lw_tr,
                        "lw_test/epoch": w_epoch,
                    }
                )
        print("")
        print("FINISHED Linear Warmup Training")
        print("")

        for i, par in enumerate(self.model.backbone.parameters()):  # type: ignore
            par.requires_grad = par_req_grad[i]  # type: ignore

        return self.model
