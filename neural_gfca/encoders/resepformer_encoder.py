from einops import rearrange, reduce, repeat
from einops._torch_specific import allow_ops_in_compiled_graph

import torch  # noqa
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as fn

from einops.layers.torch import Rearrange, Reduce

from neural_gfca.encoders.resepformer import RESepFormerBlock
from neural_gfca.nn import FlipedMakeChunk, OverlappedAdd, PositionalEncoding, SpatialPositionalEncoding

allow_ops_in_compiled_graph()


class ReSepFormerModule(nn.Module):
    def __init__(
        self,
        n_src: int,
        max_n_mic: int,
        n_stft: int,
        chunk_size: int,
        step_size: int,
        n_layers: int,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
        use_r: bool = True,
        spec_aug: nn.Module | None = None,
    ):
        super().__init__()

        if spec_aug is not None:
            self.spec_aug = nn.Sequential(
                Rearrange("b f m t -> b m f t"),
                spec_aug,
                Rearrange("b m f t -> b f m t"),
            )
        else:
            self.spec_aug = nn.Identity()

        self.bn0 = nn.BatchNorm2d(n_stft)
        self.lin0 = nn.Sequential(
            Rearrange("b c m t -> (b m) t c"),
            nn.Linear(n_stft + n_src, d_model),
            PositionalEncoding(d_model, max_len=10000),
        )

        self.lin0_post = nn.Sequential(
            SpatialPositionalEncoding(max_n_mic, d_model),
            Rearrange("b m t c -> (b m) t c"),
            FlipedMakeChunk(chunk_size, step_size),  # [B, S, T, C]
        )

        self.lin1 = nn.Linear(2 * d_model, d_model)

        self.enc = RESepFormerBlock(n_layers, d_model, n_head, dim_feedforward, norm_first, layer_norm_eps)
        self.enc.compile()

        self.use_r = use_r
        if use_r:
            self.overlapped_add = OverlappedAdd(chunk_size, step_size)
            self.head_r = nn.Sequential(
                nn.Linear(d_model, n_stft),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor, act: torch.Tensor, h0: torch.Tensor | None = None):
        B, _, M, T = x.shape

        logx = self.bn0(x.clip(1e-6).log())
        logx = self.spec_aug(logx) if self.training else logx

        h = self.lin0(torch.concat([logx, repeat(act, "b n t -> b n m t", m=M)], dim=1))
        h = self.lin0_post(rearrange(h, "(b m) t c -> b m t c", m=M))

        if h0 is None:
            h0 = torch.zeros_like(h)

        h = self.enc(self.lin1(torch.concat((h, h0), dim=-1)), n_mic=M)

        if self.use_r:
            return h + h0, rearrange(self.head_r(self.overlapped_add(h, T)), "(b m) t f -> b f m t", m=M)
        else:
            return h + h0, None


class RESepFormerEncoder(nn.Module):
    def __init__(
        self,
        n_fft: int,
        n_src: int,
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
        super().__init__()

        n_stft = n_fft // 2 + 1

        self.n_blocks = n_blocks
        self.n_src = n_src

        self.tf = nn.ModuleList()
        for _ in range(n_blocks):
            self.tf.append(
                ReSepFormerModule(
                    n_src,
                    max_n_mic,
                    n_stft,
                    chunk_size,
                    step_size,
                    n_layers,
                    d_model,
                    n_head,
                    dim_feedforward,
                    norm_first,
                    layer_norm_eps,
                    _ < n_blocks - 1,
                    spec_aug,
                )
            )

        self.diagonalizer = torch.jit.script(diagonalizer)  # type: ignore

        self.overlapped_add = OverlappedAdd(chunk_size, step_size)

        self.head_z_val = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, 2 * dim_latent),
        )

        self.head_z_att = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, n_src),
        )

        self.head_g = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            #
            Reduce("b t d -> b d", "mean"),
            #
            nn.Linear(d_model, n_stft * n_src),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, act: torch.Tensor, distribution: bool = False, Q: torch.Tensor | None = None):
        B, F, M, T = x.shape

        if Q is None:
            Q = repeat(torch.eye(M, dtype=torch.complex64, device=x.device), "m n -> b f m n", b=B, f=F)

        h, r = self.tf[0](x.abs().square(), act)
        xt = None
        for tf_ in self.tf[1:]:  # type: ignore
            Q, xt = self.diagonalizer(r, Q, x)

            h, r = tf_(xt, act, h)  # [BxM, T, C]

        h = self.overlapped_add(h, T)

        z_att = torch.softmax(reduce(self.head_z_att(h), "(b m) t n -> b m n", "mean", m=M), dim=-2)  # [B, D, N, T]

        # z
        z_mu_, z_sig_ = rearrange(self.head_z_val(h), "(b m) t (c d) -> c b d m t", m=M, c=2)  # [B, D, N, T]

        z_mu: torch.Tensor = torch.einsum("bmn,bdmt->bdnt", z_att, z_mu_)
        if distribution:  # noqa: SIM108
            qz = Normal(z_mu, fn.softplus(torch.einsum("bmn,bdmt->bdnt", z_att, z_sig_)) + 1e-6, validate_args=False)
        else:
            qz = z_mu

        # g
        g = rearrange(self.head_g(h), "(b m) (f n) -> b f m n", m=M, f=F) + 1e-6

        return qz, g, Q, xt
