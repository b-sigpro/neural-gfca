from math import ceil

from einops import rearrange, repeat
import numpy as np

import torch
from torch import nn
from torch.nn import functional as fn

from einops.layers.torch import Rearrange, Reduce


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer("pe", self.generate_encoding(d_model, max_len))

    def generate_encoding(self, d_model: int, max_len: int):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        if self.pe.shape[0] < x.size(1):
            self.pe = self.generate_encoding(self.pe.shape[1], x.size(1)).to(self.pe)

        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, max_n_mic: int, d_model: int):
        super().__init__()

        self.pe = nn.Parameter(torch.randn(1, max_n_mic, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, T, C = x.shape
        return x + self.pe[:, :M]


class TACModule(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.lin1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.PReLU(),
        )

        self.lin2 = nn.Sequential(
            Reduce("b m t c -> b t c", "mean"),
            nn.Linear(input_dim, input_dim),
            nn.PReLU(),
        )

        self.lin3 = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.PReLU(),
            Rearrange("b m t c -> (b m) t c"),
        )

    def forward(self, x: torch.Tensor, n_mic):
        """implementation of TAC

        Args:
            x (torch.Tensor, shape ``[B, T, M, C]``): input hidden vectors

        Returns:
            torch.Tensor, shape ``[B, T, M, C]``: results of TAC
        """

        h0 = rearrange(self.lin1(x), "(b m) t c -> b m t c", m=n_mic)
        h1 = repeat(self.lin2(h0), "b t c -> b m t c", m=h0.shape[1])

        return x + self.lin3(torch.concat((h0, h1), dim=-1))


class NoEinOpsTACModule(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.lin1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.PReLU(),
        )

        self.lin2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.PReLU(),
        )

        self.lin3 = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor, n_mic):
        *_, T, C = x.shape

        h0 = self.lin1(x).reshape(-1, n_mic, T, C)  # "(b m) t c -> b m t c", m=n_mic
        h1 = self.lin2(h0.mean(dim=1)).reshape(-1, 1, T, C).tile((1, n_mic, 1, 1))  # "b t c -> b m t c", m=h0.shape[1])

        return x + self.lin3(torch.concat((h0, h1), dim=-1)).reshape(-1, T, C)


class MakeChunk(nn.Module):
    def __init__(self, chunk_size: int, step_size: int):
        super().__init__()

        self.unfold = nn.Sequential(
            nn.Unfold((chunk_size, 1), stride=(step_size, 1)),
            Rearrange("b (c t) s -> b s t c", t=chunk_size),
        )

        self.chunk_size = chunk_size
        self.step_size = step_size

    def forward(self, x: torch.Tensor):
        orig_len = x.shape[-2]
        pad_len = self.chunk_size + self.step_size * (ceil((orig_len - self.chunk_size) / self.step_size)) - orig_len

        if self.chunk_size == self.step_size:
            h = fn.pad(rearrange(x, "b t c -> b c t"), (0, pad_len))
            return rearrange(h, "b c (s t) -> b s t c", t=self.chunk_size)
        else:
            return self.unfold(fn.pad(rearrange(x, "b t c -> b c t 1"), (0, 0, 0, pad_len)))


class FlipedMakeChunk(nn.Module):
    def __init__(self, chunk_size: int, step_size: int):
        super().__init__()

        self.unfold = nn.Sequential(
            nn.Unfold((chunk_size, 1), stride=(step_size, 1)),
            Rearrange("b (c t) s -> b s t c", t=chunk_size),
        )

        self.chunk_size = chunk_size
        self.step_size = step_size

    def forward(self, x: torch.Tensor):
        orig_len = x.shape[-2]
        pad_len = self.chunk_size + self.step_size * (ceil((orig_len - self.chunk_size) / self.step_size)) - orig_len

        if self.chunk_size == self.step_size:
            h = torch.concat((x_ := rearrange(x, "b t c -> b c t"), x_.flip(dims=(-1,))[..., :pad_len]), dim=-1)
            return rearrange(h, "b c (s t) -> b s t c", t=self.chunk_size)
        else:
            return self.unfold(fn.pad(rearrange(x, "b t c -> b c t 1"), (0, 0, 0, pad_len)))


class OverlappedAdd(nn.Module):
    def __init__(self, chunk_size: int, step_size: int):
        super().__init__()

        self.chunk_size = chunk_size
        self.step_size = step_size

    def fold(self, x: torch.Tensor, seq_len: int):
        h = rearrange(x, "b s t c -> b (c t) s")
        return fn.fold(h, (seq_len, 1), (self.chunk_size, 1), stride=(self.step_size, 1))

    def forward(self, x: torch.Tensor, orig_seq_len: int | None = None) -> torch.Tensor:
        """forward function

        Args:
            x (torch.Tensor): (B, S, T', C)
            orig_seq_len (int): sequence length

        Returns:
            torch.Tensor: (B, T, C)
        """

        if self.chunk_size == self.step_size:
            return rearrange(x, "b s t c -> b (s t) c")[:, :orig_seq_len]
        else:
            B, S, T, C = x.shape

            seq_len = self.chunk_size + self.step_size * (S - 1)

            x_ = self.fold(x, seq_len).squeeze(-1)
            ones_ = self.fold(torch.ones_like(x), seq_len).squeeze(-1)

            return rearrange(x_ / ones_, "b c t -> b t c")[:, :orig_seq_len]


class SDRLoss(nn.Module):
    def __init__(self, min_sdr: float = -30, eps1: float = 1e-6, eps2: float = 1e-5):
        super().__init__()

        self.min_sdr_pwr = 10 ** (min_sdr / 10)
        self.eps1 = eps1
        self.eps2 = eps2

    def forward(self, wav_est: torch.Tensor, wav_src: torch.Tensor):
        wav_src = wav_src - wav_src.mean(dim=-1, keepdim=True)
        wav_est = wav_est - wav_est.mean(dim=-1, keepdim=True)

        a = (wav_src * wav_est).mean(dim=-1, keepdim=True) / wav_src.square().mean(dim=-1, keepdim=True).clip(self.eps1)
        wav_src = a * wav_src

        s_pwr = wav_src.square().mean(dim=-1)
        e_pwr = (wav_est - wav_src).square().mean(dim=-1)

        sdr_loss_each = -10 * torch.log10(s_pwr / (e_pwr + self.min_sdr_pwr * s_pwr + self.eps2) + self.eps2)

        return sdr_loss_each
