from collections import defaultdict
from dataclasses import dataclass
import warnings

from einops import rearrange
import numpy as np

import torch
from torch import distributed as dist
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as fn  # noqa

# from ci_sdr.pt import ci_sdr_loss
from einops.layers.torch import Rearrange
from torchaudio.transforms import InverseSpectrogram, Spectrogram

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule

from neural_gfca.nn import SDRLoss

warnings.filterwarnings(
    "ignore",
    "It is recommended to use `self.log\\(.*, ..., sync_dist=True\\)`"
    " when logging on epoch level in distributed setting to accumulate the metric across"
    " devices.",
)


@dataclass
class DumpData:
    data_name: str
    x: torch.Tensor
    lm: torch.Tensor
    z: torch.Tensor
    w: torch.Tensor
    xt: torch.Tensor
    act: torch.Tensor


class UniversalTask(OptimizerLightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        n_fft: int,
        hop_length: int,
        n_src: int,
        n_noi: int,
        beta: float,
        gamma: float,
        eta: float,
        optimizer_config: OptimizerConfig,
        eps_Q: float = 1e-5,
    ):
        super().__init__(optimizer_config)

        self.encoder = encoder
        self.decoder = decoder

        self.stft = nn.Sequential(
            Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
            Rearrange("b m f t -> b f m t"),
        )

        self.istft = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

        self.hop_length = hop_length
        self.n_src = n_src
        self.n_noi = n_noi
        self.beta = beta
        self.gamma = gamma
        self.eta = eta

        self.eps_Q = eps_Q

        self._metrics = defaultdict(list)

        self.sdr_loss = SDRLoss()

    def log_async(self, name: str, value: torch.Tensor):
        self._metrics[name].append(value.item())

    @torch.autocast("cuda", enabled=False)
    def training_step(self, batch: tuple[str, dict[str, torch.Tensor]], batch_idx, log_prefix: str = "training"):
        if hasattr(self, "dump"):
            del self.dump

        data_name, data = batch
        is_supervised = data_name.startswith("sup")

        wav_mix = data["wav_mix"]
        act = data["act"]

        # make u_src
        u_src = rearrange(act, "b n (t s) -> b n t s", s=self.hop_length).amax(dim=-1)  # [B, N, T]

        # stft
        x = self.stft(wav_mix)[..., : u_src.shape[-1]]  # [B, F, M, T]
        x /= x.abs().square().mean(dim=(1, 2, 3), keepdims=True).sqrt().clip(1e-6)
        B, F, M, T = x.shape
        FMT = F * T * M

        # make u
        u = torch.concat([u_src, torch.ones([B, self.n_noi, T], device=u_src.device)], dim=1)

        # encode
        qz, g, Q, xt = self.encoder(x, u, distribution=True)
        z = qz.rsample()  # [B, D, N, T]
        _, D, *_ = z.shape

        # decode
        lm = self.decoder(z)  # [B, F, N, T]

        # check nan
        if lm.isnan().any():
            loss = torch.tensor(np.nan, device=lm.device)

            self.log_async(f"{log_prefix}/loss", loss)
            self.log_async(f"{log_prefix}/{data_name}/loss", loss)

            return self._check_nan(loss)

        # permute
        if is_supervised:
            wav_src = data["wav_src"]
            wav_src /= wav_src.square().mean(dim=-1, keepdim=True).sqrt().clip(1e-6)
            wav_src += 1e-6 * torch.randn_like(wav_src)

            qi = torch.linalg.inv(Q + self.eps_Q * torch.eye(M, device=Q.device))[:, :, 0]

        # calculate nll for x
        _, ldQ = torch.linalg.slogdet(Q)  # [B, F]

        yt = torch.einsum("bnt,bfnt,bfmn->bfmt", u, lm, g) + 1e-6
        yt = yt * torch.mean(xt.clip(1e-6) / yt, dim=3, keepdim=True)

        nll_x = yt.log().sum(dim=(1, 2, 3)) / FMT
        nll_x = nll_x + torch.sum(xt.clip(1e-6) / yt, dim=(1, 2, 3)) / FMT
        nll_x = nll_x - 2 * T * ldQ.sum(dim=1) / FMT

        # calculate kl
        kl = kl_divergence(qz, Normal(0, 1)).sum(dim=(1, 2, 3)) / FMT

        n_src = u.sum(dim=-1).count_nonzero(dim=-1)
        loss = torch.mean(n_src * (nll_x + self.beta * kl))

        # calculate supervised loss
        if is_supervised:
            Qx_yt = torch.einsum("bfmn,bfnt->bfmt", Q, x) / yt
            s = torch.einsum("bfm,bnt,bfnt,bfmn,bfmt->bnft", qi, u, lm, g, Qx_yt)

            wav_est = self.istft(s[:, : -self.n_noi], length=wav_src.shape[-1])

            sdr_loss_each = self.sdr_loss(wav_est, wav_src)

            src_mask = u[:, : -self.n_noi].sum(-1).gt(0)
            sdr_loss = (sdr_loss_each.masked_fill(~src_mask, 0).sum(1) / src_mask.sum(1)).mean()

            loss = loss + self.eta * sdr_loss

            self.log_async(f"{log_prefix}/{data_name}/sdr_loss", sdr_loss)

        # logging
        self.log_async(f"{log_prefix}/loss", loss)
        self.log_async(f"{log_prefix}/{data_name}/loss", loss)
        self.log_async(f"{log_prefix}/{data_name}/nll", nll_x.mean())
        self.log_async(f"{log_prefix}/{data_name}/kl", kl.mean())

        self.dump = DumpData(
            data_name=data_name,
            x=x[0, ..., 0, :].detach(),
            lm=lm[0].detach(),
            z=qz.mean[0].detach(),
            w=u[0].detach(),
            xt=xt[0].detach(),
            act=u[0].detach(),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, log_prefix="validation")

    def on_train_epoch_end(self):
        gatherd_metrics: list[defaultdict[str, list[float]]] = [None for _ in range(dist.get_world_size())]  # type: ignore
        dist.all_gather_object(gatherd_metrics, self._metrics)

        metrics = defaultdict(list)
        for metrics_ in gatherd_metrics:
            for k, v in metrics_.items():
                metrics[k] += v

        metrics_avg = {k: np.mean(v) for k, v in metrics.items()}

        self.log_dict(
            metrics_avg | {"step": float(self.trainer.current_epoch)},
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            sync_dist=False,
        )

        self._metrics.clear()
