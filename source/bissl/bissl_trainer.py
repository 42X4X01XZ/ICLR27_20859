from typing import Optional, Iterator, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from source.bissl.ij_solver_cg import (
    IJGradCalc,
)

from source.bissl.bisslight_trainer import (
    BiSSLightTrainerMFAC,
)


# We subclass the remake of the original BiSSL trainer of the new M-FAC trainer
# as it contains some useful functionality we wish to transfer to this trainer,
# e.g. the enhanced compatibility with LoRA.
class BiSSLTrainerOriginalViTSetting(BiSSLightTrainerMFAC):
    def __init__(
        self,
        ij_grad_calc: IJGradCalc,
        *args,
        **kwargs,
    ):
        super().__init__(ij_grad_calc=ij_grad_calc, *args, **kwargs)  # type: ignore

        del (
            self.num_lower_grads_store,
            self.lower_grads_to_upper,
            self.ij_grad_calc_device,
        )  # We don't need these attributes here, as we don't use MFAC for IJ approximation in this trainer.

    def _get_upper_grads(  # type: ignore
        self,
        upper_input: Tuple[torch.Tensor, torch.Tensor],
        lower_input: torch.Tensor,
        lam: float | torch.Tensor,
    ) -> Tuple[Iterator[torch.Tensor], Iterator[torch.Tensor], float]:

        with self.amp_context:
            loss_uhead_lbackbone = self.upper_criterion_ext_backbone(
                upper_input, ext_backbone=self.models["lower"].backbone  # type: ignore
            )

            # Concatenate the parameters of the upper head and lower backbone for gradient computation
            lower_bb_params = tuple(self._get_params_req_grad(self.models["lower"].backbone.parameters()))  # type: ignore
            upper_head_params = tuple(self._get_params_req_grad(self.models["upper"].head.parameters()))  # type: ignore

            all_grads = torch.autograd.grad(
                loss_uhead_lbackbone,
                upper_head_params + lower_bb_params,
            )

            grads_head = all_grads[: len(upper_head_params)]
            grads_backbone = all_grads[len(upper_head_params) :]

        loss_val = self._accumulate_tensor(loss_uhead_lbackbone).item()
        del loss_uhead_lbackbone  # Free the graph

        grads_backbone = tuple(grad.float() for grad in grads_backbone)
        del all_grads  # free original (half-prec) backbone grads; backbone slice no longer referenced

        ij_grads_backbone = self.ij_grad_calc(
            lower_input=lower_input,  # type: ignore
            vecs=grads_backbone,
            lam=lam,  # type: ignore
        )
        del grads_backbone  # vecs no longer needed after MFAC produces ij_grads_backbone

        torch.cuda.empty_cache()

        return (
            iter(ij_grads_backbone),
            iter(grad.float() for grad in grads_head),
            loss_val,
        )

    def _upper_train_loop(  # type: ignore
        self,
        dataloader: DataLoader | _BaseDataLoaderIter,
        lam: float | torch.Tensor,
        epoch: int,
        lower_input: torch.Tensor,
    ) -> None:

        self.models["upper"].train()
        self.models["lower"].train()

        if self.input_processors["lower"] is not None and lower_input is not None:
            lower_input = self.input_processors["lower"](lower_input)

        for upper_input in dataloader:
            self.optimizers["upper"].zero_grad(set_to_none=True)
            self.optimizers["lower"].zero_grad(set_to_none=True)

            if self.input_processors["upper"] is not None:
                upper_input = self.input_processors["upper"](upper_input)

            grads_d_ij_backbone, grads_d_ij_head, loss_uh_lbb = self._get_upper_grads(
                upper_input=upper_input,
                lower_input=lower_input,
                lam=lam,
            )
            self.loss_d_ij_avg += loss_uh_lbb

            with self.amp_context:
                loss_d_classic = self.upper_criterion_ext_backbone(upper_input)

                bb_params_upper = tuple(self._get_params_req_grad(self.models["upper"].backbone.parameters()))  # type: ignore
                head_params_upper = tuple(self._get_params_req_grad(self.models["upper"].head.parameters()))  # type: ignore

                all_classic_grads = torch.autograd.grad(
                    loss_d_classic,
                    bb_params_upper + head_params_upper,
                )
                grads_d_classic_bb = iter(
                    grad.float() for grad in all_classic_grads[: len(bb_params_upper)]
                )
                grads_d_classic_head = iter(
                    grad.float() for grad in all_classic_grads[len(bb_params_upper) :]
                )
                del all_classic_grads  # free container; individual tensors freed one-by-one as generators are consumed

            self.loss_d_classic_avg += self._accumulate_tensor(loss_d_classic).item()
            del loss_d_classic
            del upper_input  # free image batch; all forward passes using it are done

            grads_d_head = iter(
                [
                    grad_ig.add_(grad_classic)
                    for grad_ig, grad_classic in zip(
                        grads_d_ij_head, grads_d_classic_head
                    )
                ]
            )

            grads_d_backbone = iter(
                grad_ig.add_(grad_classic)
                for grad_ig, grad_classic in zip(
                    grads_d_ij_backbone, grads_d_classic_bb
                )
            )

            self._update_grad(
                grads=grads_d_backbone,
                params=self._get_params_req_grad(self.models["upper"].backbone.parameters()),  # type: ignore
            )
            self._update_grad(
                grads=grads_d_head,
                params=self._get_params_req_grad(self.models["upper"].head.parameters()),  # type: ignore
            )

            torch.nn.utils.clip_grad_norm_(self._get_params_req_grad(self.models["upper"].parameters()), 10.0)  # type: ignore

            self.optimizers["upper"].step()
            self.dataloader_step["upper"] += 1

            if self.lr_schedulers["upper"] is not None:
                self.lr_schedulers["upper"].step()

            if self.dataloader_step["upper"] % self.num_iters["upper"] == 0:
                self.upper_log(
                    loss_ig=self.loss_d_ij_avg / self.num_iters["upper"],
                    loss_classic=self.loss_d_classic_avg / self.num_iters["upper"],
                    epoch=epoch,
                )

                self.optimizers["upper"].zero_grad(set_to_none=True)
                self.optimizers["lower"].zero_grad(set_to_none=True)

                break

    def _lower_train_loop(
        self,
        dataloader: DataLoader | _BaseDataLoaderIter,
        lam: float | torch.Tensor,
        epoch: int,
    ) -> None:
        self.models["lower"].train()
        self.optimizers["lower"].zero_grad(set_to_none=True)

        for inp in dataloader:
            if self.input_processors["lower"] is not None:
                inp = self.input_processors["lower"](inp)

            self.optimizers["lower"].zero_grad(set_to_none=True)

            with self.amp_context:
                # Pretext Task Loss
                loss_p = self.models["lower"](inp)

            loss_p.backward()

            self.loss_p_avg += self._accumulate_tensor(loss_p).item()
            del loss_p

            with self.amp_context:
                # Reg loss
                reg_loss = self._regularization_loss(
                    model_pars=self._get_params_req_grad(self.models["lower"].backbone.parameters()),  # type: ignore
                    w_0=self._get_params_req_grad(self.models["upper"].backbone.parameters()),  # type: ignore
                    lam=lam,  # type: ignore
                )

                reg_grads = iter(
                    torch.autograd.grad(
                        reg_loss,
                        self._get_params_req_grad(self.models["lower"].backbone.parameters()),  # type: ignore
                    )
                )

            self.loss_reg_avg += self._accumulate_tensor(reg_loss).item()
            del reg_loss

            reg_grads = iter(grad.float() for grad in reg_grads)

            self._update_grad(
                grads=reg_grads,
                params=self._get_params_req_grad(self.models["lower"].backbone.parameters()),  # type: ignore
            )

            torch.nn.utils.clip_grad_norm_(self._get_params_req_grad(self.models["lower"].parameters()), 10.0)  # type: ignore

            self.optimizers["lower"].step()

            self.dataloader_step["lower"] += 1

            if self.lr_schedulers["lower"] is not None:
                self.lr_schedulers["lower"].step()

            if self.dataloader_step["lower"] % self.num_iters["lower"] == 0:
                self.lower_log(
                    self.loss_p_avg / self.num_iters["lower"],
                    self.loss_reg_avg / self.num_iters["lower"],
                    lam=lam,  # type: ignore
                    epoch=epoch,
                )

                self.optimizers["lower"].zero_grad(set_to_none=True)

                del inp  # free last batch before upper level
                break

    def train_one_epoch(
        self,
        lam: float | torch.Tensor = 1.0,
        epoch: Optional[int] = None,
    ):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        self.models["upper"].train()
        self.models["lower"].train()

        #########################
        ###### Lower Level ######
        #########################
        print("Lower Level...")

        self._calibrate_dataloader("lower")
        self.loss_p_avg, self.loss_reg_avg = 0.0, 0.0
        if self.num_iters["lower"] > 0:
            self._train_one_level("lower", trainer_fn=self._lower_train_loop, lam=lam)

        torch.cuda.empty_cache()

        print("")

        ##########################
        ####### Upper Level ######
        ##########################
        print("Upper Level...")

        lower_input = next(iter(self.dataloaders["lower"]))
        self._calibrate_dataloader("upper")

        torch.cuda.empty_cache()

        self.loss_d_ij_avg, self.loss_d_classic_avg = 0.0, 0.0
        if self.num_iters["upper"] > 0:
            self._train_one_level(
                "upper",
                trainer_fn=self._upper_train_loop,
                lam=lam,
                lower_input=lower_input,
            )

        torch.cuda.empty_cache()
