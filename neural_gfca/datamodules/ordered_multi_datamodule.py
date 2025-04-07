from typing import Any

from collections.abc import Callable, Iterable

import numpy as np

from torch.utils.data import DataLoader, Dataset

import lightning as lt
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from aiaccel.torch.datasets import CachedDataset, scatter_dataset
import yaml


def infinite_generator(data_loader):
    while True:
        yield from data_loader


class RandomCombinedLoader(Iterable):
    def __init__(self, dataloaders: dict[str, Any], steps: int):
        self.dataloaders = dataloaders
        self.steps = steps

        self.rng = np.random.Generator(np.random.PCG64(0))

    def __iter__(self):
        key_list = list(self.dataloaders)
        for _ in range(self.steps):
            key = key_list[self.rng.choice(len(key_list))]
            yield key, next(self.dataloaders[key])

    def __len__(self):
        return self.steps


class RandomMultiDataModule(lt.LightningDataModule):
    def __init__(
        self,
        train_dataset_fn_dict: dict[str, Callable[..., Dataset[str]]],
        val_dataset_fn_dict: dict[str, Callable[..., Dataset[str]]],
        batch_size: int,
        train_steps: int,
        val_steps: int,
        use_cache: bool = False,
        use_scatter: bool = True,
        num_workers: int = 10,
        common_args: dict[str, Any] | None = None,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.train_dataset_fn_dict = train_dataset_fn_dict
        self.val_dataset_fn_dict = val_dataset_fn_dict
        self.common_args = common_args if common_args is not None else dict()

        self.batch_size = batch_size
        self.train_steps = train_steps
        self.val_steps = val_steps

        self.use_cache = use_cache
        self.use_scatter = use_scatter

        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str | None):
        if stage == "fit":
            self.train_dataset_dict = {key: fn(**self.common_args) for key, fn in self.train_dataset_fn_dict.items()}
            self.val_dataset_dict = {key: fn(**self.common_args) for key, fn in self.val_dataset_fn_dict.items()}

            dataset_stats = {
                "Training set": {k: len(v) for k, v in self.train_dataset_dict.items()},  # type: ignore
                "Validation set": {k: len(v) for k, v in self.val_dataset_dict.items()},  # type: ignore
            }
            rank_zero_info(yaml.dump(dataset_stats))

            if self.use_cache:
                self.train_dataset_dict = {key: CachedDataset(ds) for key, ds in self.train_dataset_dict.items()}
                self.val_dataset_dict = {key: CachedDataset(ds) for key, ds in self.val_dataset_dict.items()}

            if self.use_scatter:
                self.train_dataset_dict = {key: scatter_dataset(ds) for key, ds in self.train_dataset_dict.items()}
                self.val_dataset_dict = {key: scatter_dataset(ds) for key, ds in self.val_dataset_dict.items()}
        else:
            raise ValueError("`stage` is not 'fit'.")

    def _create_dataloader(self, dataset_dict: dict[str, Any], batch_size: int, steps: int, **kwargs: Any):
        dataloaders = {}
        for key, ds in dataset_dict.items():
            dataloaders[key] = infinite_generator(
                DataLoader(
                    dataset=ds,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    shuffle=True,
                    pin_memory=self.pin_memory,
                    **kwargs,
                )
            )

        return RandomCombinedLoader(dataloaders, steps)

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset_dict, self.batch_size, self.train_steps, drop_last=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset_dict, self.batch_size, self.val_steps, drop_last=False)
