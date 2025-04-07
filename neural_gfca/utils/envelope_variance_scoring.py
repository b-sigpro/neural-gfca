from einops import reduce

import torch
from torch import nn

from torchaudio.transforms import MelSpectrogram


class EnvelopeVarianceScoring(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 200, n_mels: int = 40):
        super().__init__()
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2,
        )

        self.hop_length = hop_length

    def forward(self, src_wav: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        u = reduce(
            mask[: mask.shape[0] // self.hop_length * self.hop_length],
            "(t s) -> t",
            "max",
            s=self.hop_length,
        )

        logx = self.mel(src_wav)[..., : u.shape[-1]].add(1e-6).log()[..., u > 0.5]  # [M, F, T]

        x = torch.exp(logx - logx.mean(dim=-1, keepdim=True))

        var = torch.var(x ** (1 / 3), dim=-1)  # [M, F]
        score = torch.sum(var / torch.amax(var, dim=0, keepdim=True), dim=-1)  # [M]

        indices = score.argsort(descending=True)

        return indices
