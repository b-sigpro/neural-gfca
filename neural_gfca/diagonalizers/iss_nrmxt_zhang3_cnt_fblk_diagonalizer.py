import torch
from torch import nn


class ISSDiagonalizer(nn.Module):
    def __init__(
        self, n_iter: int = 1, eps: float = 1e-6, eps2: float = 1e-6, eps3: float = 1e-3, freq_batch: int = 50
    ):
        super().__init__()

        self.n_iter = n_iter

        self.eps = eps
        self.eps2 = eps2
        self.eps3 = eps3

        self.freq_batch = freq_batch

    def forward(self, r, Q, x):
        """
        Parameters
        ----------
        r : (B, F, M, T) Tensor
        Q : (B, F, M, M) Tensor
        x : (B, F, M, T) Tensor
        """

        _, F, M, T = x.shape

        r = r.contiguous()
        x = x.contiguous()

        for f in range(0, F, self.freq_batch):
            r_ = r[:, f : f + self.freq_batch].clone()
            Q_ = Q[:, f : f + self.freq_batch].clone()
            x_ = x[:, f : f + self.freq_batch].clone()

            V = torch.einsum("...kt,...mt,...nt->...kmn", r_.clip(self.eps3), x_, x_.conj()) / T
            trV = torch.einsum("...mm->...", V)[..., None, None].real
            V = V + trV.maximum(torch.ones_like(trV)).to(V.dtype) * self.eps * torch.eye(M, device="cuda")

            for _ in range(self.n_iter):
                for k in range(M):
                    q = Q_[..., k, :]
                    Vq = torch.einsum("...kmn,...n->...km", V, q.conj())

                    qVq = torch.einsum("...m,...km->...k", q, Vq).real.clip(self.eps2)
                    v = torch.einsum("...km,...km->...k", Q_, Vq) / qVq.to(x_.dtype)
                    v[..., k] = 1 - qVq[..., k] ** -0.5

                    Q_ = Q_ - torch.einsum("...m,...n->...mn", v, q)

            Q = Q.clone()
            Q[:, f : f + self.freq_batch] = Q_

        Qx = torch.einsum("...mn,...nt->...mt", Q, x)
        xt = Qx.real**2 + Qx.imag**2  # torch.abs(Qx) ** 2

        scale = xt.mean(dim=(1, 2, 3), keepdim=True)
        xt = xt / scale
        Q = Q / scale.clip(1e-6).sqrt().to(x.dtype)

        return Q, xt
