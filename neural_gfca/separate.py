from argparse import Namespace
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc

from einops import rearrange, reduce

import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as fn  # noqa

from torchaudio.transforms import InverseSpectrogram

import soundfile as sf

from neural_gfca.bf.mvdr import mvdr_ban
from neural_gfca.utils.envelope_variance_scoring import EnvelopeVarianceScoring
from neural_gfca.utils.separator import main


@dataclass
class Context:
    model: nn.Module
    istft: nn.Module
    config: ListConfig | DictConfig
    evs: EnvelopeVarianceScoring


def add_common_args(parser):
    parser.add_argument("--out_ch", type=int, default=0)
    parser.add_argument("--n_mic", type=int, default=-1)
    parser.add_argument("--normalize", type=str, default="none")
    parser.add_argument("--target", action="store_true")
    parser.add_argument("--use_mvdr", action="store_true")
    parser.add_argument("--drop_context", action="store_true")
    parser.add_argument("--mvdr_eps", type=float, default=1e-8)
    parser.add_argument("--noi_snr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")


def initialize(args: Namespace, unk_args: list[str]):
    torch._dynamo.config.suppress_errors = True

    config = oc.merge(
        oc.load(args.model_path / "config.yaml"),  # todo
        oc.from_cli(unk_args),
    )

    checkpoint_path = args.model_path / "version_0" / "checkpoints" / "last.ckpt"
    config.task._target_ += ".load_from_checkpoint"
    model = instantiate(
        config.task,
        checkpoint_path=checkpoint_path,
        map_location=args.device,
    )
    model.eval()

    istft = InverseSpectrogram(model.stft[0].n_fft, hop_length=model.stft[0].hop_length).to(args.device)

    evs = EnvelopeVarianceScoring().to(args.device)

    ctx = Context(model, istft, config, evs)

    return ctx


def separate(src_filename: Path, dst_filename: Path, ctx: Context, args: Namespace, unk_args: list[str]):
    # load info
    with open(src_filename.with_suffix(".info"), "rb") as f:
        info = pkl.load(f)

    act = torch.tensor(info["act"], dtype=torch.float32, device=args.device)
    start = info["start"]
    end = info["end"]

    # create u
    u_src = reduce(
        act[: act.shape[0] // ctx.config.hop_length * ctx.config.hop_length],
        "(t s) n -> 1 n t",
        "max",
        s=ctx.config.hop_length,
    )

    if u_src.shape[1] > (n_src_ := ctx.config.n_src - ctx.config.n_noi):
        s_indices = torch.zeros(n_src_, dtype=torch.long)
        s_indices[1:n_src_] = u_src[:, 1:].sum(dim=(0, 2)).argsort(descending=True)[: n_src_ - 1].sort().values + 1

        u_src = u_src[:, s_indices]

    u = torch.zeros([1, ctx.config.n_src, u_src.shape[-1]], device=args.device)
    u[:, : u_src.shape[1]] = u_src
    u[:, -ctx.config.n_noi :] = 1

    # load wav
    src_wav, sr = sf.read(src_filename)
    src_wav = torch.tensor(src_wav.T, dtype=torch.float32, device=args.device)

    if args.n_mic > 0:
        mic_indices = ctx.evs(src_wav, act[:, 0])
        src_wav = src_wav[mic_indices[: args.n_mic]]

    # calculate x
    xraw = ctx.model.stft(src_wav.unsqueeze(0))[..., : u.shape[-1]]  # [B, F, M, T]

    scale = xraw.abs().square().clip(1e-6).mean(dim=(1, 2, 3), keepdims=True).sqrt()
    x = xraw / scale

    # encode
    z, g, Q, _ = ctx.model.encoder(x, u)

    # decode
    lm = ctx.model.decoder(z)  # [B, F, N, T]

    # Wiener filtering
    yt = torch.einsum("bnt,bfnt,bfmn->bfmt", u, lm, g).add(1e-6)
    Qx_yt = torch.einsum("bfmn,bfnt->bfmt", Q, xraw) / yt
    s_wf = torch.einsum("bflm,bnt,bfnt,bfmn,bfmt->bnflt", torch.linalg.inv(Q), u, lm, g, Qx_yt)

    # perform mvdr-ban
    if args.use_mvdr:
        s = []
        for k in [0] if args.target else range(u.shape[1]):
            s_ = s_wf.squeeze(0)[k]
            n_ = xraw.squeeze(0) - s_

            if args.drop_context:
                start_ = floor(start * sr / ctx.config.hop_length)
                end_ = ceil(end * sr / ctx.config.hop_length)

                s_, n_ = s_[..., start_:end_], n_[..., start_:end_]

            Rs = torch.einsum("fnt,fmt->fnm", s_, s_.conj()) / s_.shape[-1]
            Rn = torch.einsum("fnt,fmt->fnm", n_, n_.conj()) / n_.shape[-1]

            h = mvdr_ban(Rs, Rn, eps=args.mvdr_eps)

            s.append(torch.einsum("fm,fmt->ft", h.conj(), xraw.squeeze(0)))

        s = torch.stack(s, dim=0)
    else:
        s = s_wf.squeeze(0)[..., args.out_ch, :]
        if args.target:
            s = s[0].unsqueeze(0)

    # reconstract time-domain signal
    dst_wav = rearrange(ctx.model.istft(s), "m t -> t m")

    if args.noi_snr is not None:
        scale = dst_wav.square().mean().sqrt().clip(1e-6) * 10 ** (-args.noi_snr / 20)
        dst_wav = dst_wav + torch.randn_like(dst_wav) * scale

    # save audio file
    if args.target:
        start_ = floor(start * sr)
        end_ = ceil(end * sr)

        dst_wav = dst_wav[start_:end_]

    assert args.normalize in ["none", "always", "exceed"]
    if args.normalize == "always" or (args.normalize == "exceed" and dst_wav.abs().amax() > 1):
        dst_wav /= dst_wav.abs().amax().clip(1e-6)

    sf.write(dst_filename, dst_wav.cpu().numpy(), sr, "PCM_24")


if __name__ == "__main__":
    main(add_common_args, initialize, separate)
