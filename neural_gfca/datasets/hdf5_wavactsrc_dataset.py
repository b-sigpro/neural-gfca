from typing import Literal

from pathlib import Path
from warnings import warn

import numpy as np

import torch
from torch.utils.data import Dataset

from aiaccel.torch.datasets import CachedDataset, FileCachedDataset, HDF5Dataset


class HDF5WavActSrcDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path | str,
        grp_list: Path | str | list[str] | None = None,
        num_speakers: int = -1,
        num_channels: int = -1,
        channel_selection: Literal["none", "power", "envelope", "random"] = "none",
        randperm_mic: bool = True,
        sort_spk: bool = True,
        randperm_spk: bool = False,
        use_cache: bool | str = True,
        duration: int | None = None,
        sr: int | None = None,
        cache_path: Path | None = None,
    ) -> None:
        super().__init__()

        self._dataset = HDF5Dataset(dataset_path, grp_list)
        if use_cache is True or use_cache == "mem":
            self._dataset = CachedDataset(self._dataset)
        elif use_cache == "file":
            assert cache_path is not None
            self._dataset = FileCachedDataset(self._dataset, cache_path)

        self.num_channels = num_channels
        self.num_speakers = num_speakers

        self.channel_selection = channel_selection
        if channel_selection == "envelope":
            from neural_gfca.utils.envelope_variance_scoring import EnvelopeVarianceScoring

            self.evs = EnvelopeVarianceScoring()

        self.randperm_mic = randperm_mic

        self.sort_spk = sort_spk
        self.randperm_spk = randperm_spk

        self.duration = duration
        self.sr = sr

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        item = self._dataset[index]
        wav_mix = item["wav"]
        wav_src = item["src"]
        act = item["act"].to(torch.float32)

        if self.duration is not None:
            duration = self.sr * self.duration
            t_start = np.random.randint(0, act.shape[1] - duration + 1)
            t_end = t_start + duration

            act = act[:, t_start:t_end]

            wav_mix = wav_mix[:, t_start:t_end]
            wav_src = wav_src[..., t_start:t_end]

        if self.num_channels > 0:
            match self.channel_selection:
                case "power":
                    m_indices = torch.mean(wav_mix**2, dim=-1).argsort(dim=0, descending=True)[: self.num_channels]
                case "envelope":
                    act_ = act.amax(dim=0)
                    if act_.amax() == 0:
                        warn(
                            "Activation is all zero, which fallbacks the envelope selection into naive selection.",
                            stacklevel=2,
                        )
                        m_indices = np.arange(self.num_channels)
                    else:
                        m_indices = self.evs(wav_mix, act_)[: self.num_channels]
                case "random":
                    raise NotImplementedError()
                case "none":
                    m_indices = np.arange(self.num_channels)
                case _:
                    raise ValueError()

            wav_mix = wav_mix[m_indices]
            wav_src = wav_src[m_indices]

        if self.randperm_mic:
            m_indices = torch.randperm(wav_mix.shape[0])
            wav_mix = wav_mix[m_indices]
            wav_src = wav_src[m_indices]

        wav_src = wav_src[0]

        if self.num_speakers > 0:
            N, T = act.shape

            (tmp := torch.zeros((self.num_speakers, T)))[:N] = act
            act = tmp

            (tmp := torch.zeros((self.num_speakers, T)))[:N] = wav_src
            wav_src = tmp

        if self.sort_spk:
            order = act.sum(-1).argsort(descending=True)
            act = act[order]
            wav_src = wav_src[order]

        if self.randperm_spk:
            order = torch.randperm(act.shape[0])
            act = act[order]
            wav_src = wav_src[order]

        return {"wav_mix": wav_mix, "act": act, "wav_src": wav_src}
