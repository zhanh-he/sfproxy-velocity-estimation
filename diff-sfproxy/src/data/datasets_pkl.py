from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.utils.data import Dataset


class NoteProxyDatasetPkl(Dataset):
    """Dataset for loading note-conditioned proxy training data from pickled tensors."""

    def __init__(self, path_to_dataset: Union[str, Path], split: str = "all", mmap: bool = True):
        super().__init__()

        self.path_to_dataset = Path(path_to_dataset)
        if not self.path_to_dataset.is_dir():
            raise ValueError(f"{self.path_to_dataset} is not a directory")

        with open(self.path_to_dataset / "configs.pkl", "rb") as f:
            self.configs_dict: Dict = torch.load(f)

        assert split in ["train", "test", "all"], f"Unknown split: {split}"
        self.split = split
        self.is_mmap = mmap

        self.inputs_pitch = None
        self.inputs_cont = None
        self.inputs_mask = None
        self.targets_note = None
        self.targets_seg = None

    @property
    def name(self) -> str:
        return self.path_to_dataset.stem

    @property
    def instrument_name(self) -> str:
        return str(self.configs_dict.get("instrument", "unknown"))

    @property
    def sr(self) -> int:
        return int(self.configs_dict["sr"])

    @property
    def seg_len_s(self) -> float:
        return float(self.configs_dict["seg_len_s"])

    @property
    def nmax(self) -> int:
        return int(self.configs_dict["nmax"])

    @property
    def d_note(self) -> int:
        return int(self.configs_dict["d_note"])

    @property
    def d_seg(self) -> Optional[int]:
        v = self.configs_dict.get("d_seg")
        return None if v is None else int(v)

    def __len__(self) -> int:
        if self.split == "all":
            return int(self.configs_dict["dataset_size"])
        if self.split == "train":
            return int(self.configs_dict["train_size"])
        if self.split == "test":
            return int(self.configs_dict["test_size"])
        return 0

    def __getitem__(self, index: int):
        if self.inputs_pitch is None:
            self._load_dataset()

        batch = {
            "pitch": self.inputs_pitch[index],
            "cont": self.inputs_cont[index],
            "mask": self.inputs_mask[index],
            "target_note": self.targets_note[index],
        }

        if self.targets_seg is not None:
            batch["target_seg"] = self.targets_seg[index]

        return batch

    def _load_dataset(self) -> None:
        suffix = "_" + self.split if self.split != "all" else ""

        self.inputs_pitch = torch.load(
            str(self.path_to_dataset / f"inputs_pitch{suffix}.pkl"), map_location="cpu", mmap=self.is_mmap
        )
        self.inputs_cont = torch.load(
            str(self.path_to_dataset / f"inputs_cont{suffix}.pkl"), map_location="cpu", mmap=self.is_mmap
        )
        self.inputs_mask = torch.load(
            str(self.path_to_dataset / f"inputs_mask{suffix}.pkl"), map_location="cpu", mmap=self.is_mmap
        )
        self.targets_note = torch.load(
            str(self.path_to_dataset / f"targets_note{suffix}.pkl"), map_location="cpu", mmap=self.is_mmap
        )

        # Optional
        p = self.path_to_dataset / f"targets_seg{suffix}.pkl"
        if p.exists():
            self.targets_seg = torch.load(str(p), map_location="cpu", mmap=self.is_mmap)

        # Basic sanity
        n = self.inputs_pitch.shape[0]
        assert self.inputs_cont.shape[0] == n
        assert self.inputs_mask.shape[0] == n
        assert self.targets_note.shape[0] == n
        if self.targets_seg is not None:
            assert self.targets_seg.shape[0] == n
