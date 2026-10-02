from typing import List, Tuple, Optional

import torch


class IJGradCalcMFACLW:
    def __init__(
        self,
        parameters: Tuple[torch.nn.Parameter, ...],
        param_groups_idx: Optional[List[List[int]]] = None,
    ):
        """
        Since we are approximating the inverse hessian vector product, which we know does result in a term not involving theta_D,
        i am instead choosing the vector of L^P + norm(theta_P) instead of L^P + norm(theta_P - theta_D).
        This gives the advantage of the hessian of this objective being constant during each update of the upper-level,
        making the lower-level gradients passed to this solver being more "accurate" (not totally, as they are gradients
        from ealier updates, but at least its a bit more precise.)

        For this solver, we hence assume that the pretext_grads given simply are the gradients of L^P solely, not the
        entire lower-level objective (i.e. without the reg term.)

        """

        self.parameters: Tuple[torch.nn.Parameter, ...] = parameters
        self.param_groups_idx: Optional[List[List[int]]] = param_groups_idx

        self.b = None
        self.m = None
        self.Hgs = None
        self.denoms = None
        self.device = None
        self.dtype = None

    def update(
        self,
        pretext_grads: Tuple[Tuple[torch.Tensor, ...], ...],
        lam: float | torch.Tensor,
    ) -> None:

        assert (
            isinstance(lam, (float, torch.Tensor)) and lam > 0.0
        ), "The regularization parameter 'lam' must be a positive float/Tensor."

        # Inspired by https://github.com/IST-DASLab/M-FAC/blob/master/prun.py
        # Difference: Here we consider each "batch" to be a different module
        # I.e. we have to store our values in tuples instead of tensors since
        # each module may have different shapes.

        if self.param_groups_idx is not None:
            pretext_grads = tuple(
                tuple(
                    torch.cat(
                        [grads[grp_idx].contiguous().view(-1) for grp_idx in group_idxs]
                    )
                    for group_idxs in self.param_groups_idx
                )
                for grads in pretext_grads
            )

        b = len(pretext_grads[0])
        m = len(pretext_grads)
        device = pretext_grads[0][0].device
        dtype = pretext_grads[0][0].dtype

        # Free previous epoch's Hgs/denoms before allocating new ones to avoid
        # the 2x memory peak (old + new coexisting) and the fragmentation it causes.
        self.Hgs = None
        self.denoms = None

        denoms = torch.zeros((m, b), device=device, dtype=dtype)  # m x b
        if device.type == "cpu":
            # Hgs[j]: [m, dim_j] — inner loop uses BLAS gemv instead of Python-dispatched dots
            Hgs = [
                torch.zeros(m, pretext_grads[0][j].numel(), device=device, dtype=dtype)
                for j in range(b)
            ]
            gs = [pretext_grads[0][j].contiguous().view(-1) for j in range(b)]
            for j in range(b):
                Hgs[j][0] = gs[j] / lam
                denoms[0, j] = m + torch.dot(gs[j], Hgs[j][0]).item()
            for i in range(1, m):
                gs = [pretext_grads[i][j].contiguous().view(-1) for j in range(b)]
                for j in range(b):
                    Hgs[j][i] = gs[j] / lam
                    muls_j = (Hgs[j][:i] @ gs[j]) / denoms[:i, j]
                    Hgs[j][i] -= muls_j @ Hgs[j][:i]
                    denoms[i, j] = m + torch.dot(gs[j], Hgs[j][i]).item()

        else:  # Assuming its on GPU, we can use the more efficient tensor operations

            gs = tuple(g.clone() for g in pretext_grads[0])  # b x dim_layer
            Hgs = [
                tuple(torch.zeros_like(g, device=device, dtype=dtype) for g in gs)
                for _ in range(m)
            ]  # m x b x dim_layer

            Hgs[0] = tuple((1 / lam) * g.clone() for g in gs)
            denoms[0, :] = torch.tensor(
                [m + torch.sum(g * Hg).item() for g, Hg in zip(gs, Hgs[0])],
                device=device,
                dtype=dtype,
            )

            for i in range(1, m):
                gs = tuple(g.clone() for g in pretext_grads[i])
                Hgs[i] = tuple((1 / lam) * g.clone() for g in gs)
                muls = (
                    torch.tensor(  # i x b
                        [
                            [torch.sum(g * Hg).item() for g, Hg in zip(gs, Hgs[j])]
                            for j in range(i)
                        ],
                        device=device,
                        dtype=dtype,
                    )
                    / denoms[:i, :]
                )
                Hgs[i] = tuple(
                    Hg - sum(muls[k, j] * Hgs[k][j] for k in range(i))
                    for j, Hg in enumerate(Hgs[i])
                )

                denoms[i, :] = torch.tensor(
                    [m + torch.sum(g * Hg).item() for g, Hg in zip(gs, Hgs[i])],
                    device=device,
                    dtype=dtype,
                )

        self.b = b
        self.m = m
        self.Hgs = Hgs
        self.denoms = denoms
        self.device = device
        self.dtype = dtype

    def __call__(  # type: ignore
        self,
        vecs: Tuple[torch.Tensor, ...],
        lam: float | torch.Tensor = 1.0,
    ) -> Tuple[torch.Tensor, ...]:

        assert (
            isinstance(lam, (float, torch.Tensor)) and lam > 0.0
        ), "The regularization parameter 'lam' must be a positive float/Tensor."

        assert all(
            val is not None for val in [self.b, self.m, self.Hgs, self.denoms]
        ), "MFAC Static IJ Grad Calc not initialized. Call update() first."

        assert all(vec.shape == par.shape for vec, par in zip(vecs, self.parameters))
        assert self.device is not None

        if self.param_groups_idx is not None:
            vecs = tuple(
                torch.cat(
                    [vecs[grp_idx].contiguous().view(-1) for grp_idx in group_idxs]
                )
                for group_idxs in self.param_groups_idx
            )

        vecs_device = vecs[0].device
        vecs = tuple(v.to(self.device) for v in vecs)

        if self.device.type == "cpu":
            results = []
            for j, vec in enumerate(vecs):
                vec_flat = vec.contiguous().view(-1)
                muls_j = (self.Hgs[j] @ vec_flat) / self.denoms[:, j]  # type: ignore
                result_flat = vec_flat - lam * (muls_j @ self.Hgs[j])  # type: ignore
                results.append(result_flat.view(vec.shape))
            results = tuple(results)
        else:
            muls = (
                torch.tensor(
                    [[torch.sum(hg * v) for hg, v in zip(Hg, vecs)] for Hg in self.Hgs],  # type: ignore
                    dtype=self.dtype,
                    device=self.device,
                )
                / self.denoms  # type: ignore
            )
            results = tuple(
                vec - lam * sum(muls[k, j] * self.Hgs[k][j] for k in range(self.m))  # type: ignore
                for j, vec in enumerate(vecs)
            )

        if self.param_groups_idx is not None:
            results_flat = torch.cat([result.contiguous().view(-1) for result in results])  # type: ignore

            results = []
            offset = 0

            for p in self.parameters:
                results.append(
                    results_flat[offset : offset + p.nelement()].view(p.size())
                )
                offset += p.nelement()
            del results_flat

            results = tuple(results)

        return tuple(r.to(vecs_device) for r in results)


# Global version, not used in the current implementation, but kept for reference:


class IJGradCalcMFACGlobal(IJGradCalcMFACLW):
    def update(
        self,
        pretext_grads: Tuple[Tuple[torch.Tensor, ...], ...],
        lam: float | torch.Tensor,
    ) -> None:

        assert (
            isinstance(lam, (float, torch.Tensor)) and lam > 0.0
        ), "The regularization parameter 'lam' must be a positive float/Tensor."

        # Inspired by source.MFAC.prun.HInvFastBatch
        # Here we consider the entire model as one single module

        m = len(pretext_grads)
        device = pretext_grads[0][0].device
        dtype = pretext_grads[0][0].dtype

        # Free previous epoch's Hgs/denoms before allocating new ones to avoid
        # the 2x memory peak (old + new coexisting) and the fragmentation it causes.
        self.Hgs = None
        self.denoms = None
        torch.cuda.empty_cache()

        denoms = torch.zeros((m,), device=device, dtype=dtype)  # m
        gs = torch.cat([g.clone().view(-1) for g in pretext_grads[0]])  # total_dim
        Hgs = torch.zeros((m, gs.numel()), device=device, dtype=dtype)  # m x total_dim

        Hgs[0] = (1 / lam) * gs
        denoms[0] = m + torch.sum(gs * Hgs[0]).item()

        for i in range(1, m):
            gs = torch.cat([g.clone().view(-1) for g in pretext_grads[i]])  # total_dim
            Hgs[i] = (1 / lam) * gs
            muls = (
                torch.tensor(  # i
                    [torch.sum(gs * Hgs[j]).item() for j in range(i)],
                    device=device,
                    dtype=dtype,
                )
                / denoms[:i]
            )
            Hgs[i] = Hgs[i] - sum(muls[j] * Hgs[j] for j in range(i))

            denoms[i] = m + torch.sum(gs * Hgs[i]).item()

        self.m = m
        self.Hgs = Hgs
        self.denoms = denoms
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        vecs: Tuple[torch.Tensor, ...],
        lam: float | torch.Tensor = 1.0,
    ) -> Tuple[torch.Tensor, ...]:

        assert (
            isinstance(lam, (float, torch.Tensor)) and lam > 0.0
        ), "The regularization parameter 'lam' must be a positive float/Tensor."

        assert all(
            val is not None for val in [self.m, self.Hgs, self.denoms]
        ), "MFAC Static IJ Grad Calc not initialized. Call update() first."

        assert all(vec.shape == par.shape for vec, par in zip(vecs, self.parameters))

        # total_dim
        vecs_device = vecs[0].device
        # total_dim — move to CPU; Hgs/denoms are already on CPU
        vecs_flat = torch.cat([v.contiguous().cpu().view(-1) for v in vecs])  # type: ignore

        # m — on CPU
        muls = (
            torch.tensor(
                [torch.sum(hg * vecs_flat) for hg in self.Hgs],  # type: ignore
                dtype=self.dtype,
                device=self.device,
            )
            / self.denoms  # type: ignore
        )
        # total_dim — on CPU
        result = vecs_flat - lam * sum(muls[k] * self.Hgs[k] for k in range(self.m))  # type: ignore

        # Turn the flat result back into original shape and move back to GPU
        ij_grads_backbone = []
        offset = 0

        for p in self.parameters:
            ij_grads_backbone.append(
                result[offset : offset + p.nelement()].view(p.size()).to(vecs_device)
            )
            offset += p.nelement()
        del result

        return tuple(ij_grads_backbone)
