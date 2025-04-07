from math import ceil

from einops import repeat

import torch  # noqa
from torch import nn

from neural_gfca.encoders.resepformer_encoder import RESepFormerEncoder as RESepFormerEncoder_


class RESepFormerEncoder(RESepFormerEncoder_):
    def __init__(
        self,
        n_fft: int,
        n_src: int,
        n_pre_iter: int,
        max_n_mic: int,
        dim_latent: int,
        chunk_size: int,
        step_size: int,
        diagonalizer: nn.Module,
        n_blocks: int,
        n_layers: int = 1,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
        spec_aug: nn.Module | None = None,
    ):
        super().__init__(
            n_fft,
            n_src,
            max_n_mic,
            dim_latent,
            chunk_size,
            step_size,
            diagonalizer,
            n_blocks,
            n_layers,
            d_model,
            n_head,
            dim_feedforward,
            norm_first,
            layer_norm_eps,
            spec_aug,
        )

        self.n_pre_iter = n_pre_iter

    def forward(self, x: torch.Tensor, act: torch.Tensor, distribution: bool = False):
        B, F, M, T = x.shape

        Q = repeat(torch.eye(M, dtype=torch.complex64, device=x.device), "m n -> b f m n", b=B, f=F)
        with torch.no_grad():
            lm = repeat(act, "b n t -> b f n t", f=F)
            g = repeat(torch.eye(self.n_src, device="cuda"), "m n -> (l m) n", l=ceil(M / self.n_src))[:M].clip(0.1)

            r = 1 / torch.einsum("mn,bfnt->bfmt", g, lm)

            for _ in range(self.n_pre_iter):
                Q, _ = self.diagonalizer(r, Q, x)

        return super().forward(x, act, distribution, Q)
