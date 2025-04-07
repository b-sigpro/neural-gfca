from typing import Literal

from pathlib import Path
from warnings import warn

import numpy as np

import torch
from torch.utils.data import Dataset

from aiaccel.torch.datasets import CachedDataset, FileCachedDataset, HDF5Dataset


class HDF5WavActDataset(Dataset):
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
        act = item["act"].to(torch.float32)

        if self.duration is not None:
            duration = self.sr * self.duration
            t_start = np.random.randint(0, act.shape[1] - duration + 1)
            t_end = t_start + duration

            act = act[:, t_start:t_end]

            wav_mix = wav_mix[:, t_start:t_end]

        if self.num_channels > 0:
            match self.channel_selection:
                case "power":
                    m_indices = torch.mean(wav_mix**2, dim=-1).argsort(dim=0, descending=True)[: self.num_channels]
                    wav_mix = wav_mix[m_indices]
                case "envelope":
                    act_ = act.amax(dim=0)
                    if act_.amax() == 0:
                        warn(
                            "Activation is all zero, which fallbacks the envelope selection into naive selection.",
                            stacklevel=2,
                        )
                        wav_mix = wav_mix[: self.num_channels]  # todo: assert
                    else:
                        indices = self.evs(wav_mix, act_)
                        wav_mix = wav_mix[indices[: self.num_channels]]
                case "random":
                    raise NotImplementedError()
                case "none":
                    wav_mix = wav_mix[: self.num_channels]
                case _:
                    raise ValueError()

        if self.randperm_mic:
            wav_mix = wav_mix[torch.randperm(wav_mix.shape[0])]

        if self.num_speakers > 0:
            N, T = act.shape

            (tmp := torch.zeros((self.num_speakers, T)))[:N] = act
            act = tmp

        if self.sort_spk:
            act = act[act.sum(-1).argsort(descending=True)]

        if self.randperm_spk:
            act = act[torch.randperm(act.shape[0])]

        return {"wav_mix": wav_mix, "act": act}
