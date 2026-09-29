# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch dataset class for loading saved-off conditioning/audio datasets

from globals import *
import os
import numpy as np
from torch.utils.data import Dataset
import ui

class GsetMidiDataset(Dataset):
    """
    Torch dataset class for loading saved-off conditioning/audio datasets.
    """
    def __init__(self, name=None, datasets_path=DATASETS_PATH, dtype=DEFAULT_NP_DTYPE):
        """
        Initialize parameters for loading a prepared .npz dataset.

        Parameters
        ----------
        name : string
            Name of dataset to load. If None, give dropdown of options from data_set path?
        datasets_path : string
            Path to datasets
        dtype : numpy data type
            numpy data type to load data as
        """
        if name==None:  
            list_of_avail_datasets = [dset for dset in os.listdir(datasets_path) if dset.endswith(".npz")]
            name = ui.list_selection_menu(list_of_avail_datasets)

        load_path = os.path.join(datasets_path, name)
        x = np.load(load_path)

        self.dataset = {}

        for key in x.keys():
            self.dataset[key] = x[key].astype(dtype)
        
        self.len = len(self.dataset['conditioning']) # get the length of the dataset from one of the keys

        self.name = name

    def __len__(self):
        """
        Return length of the dataset.
        """
        return self.len

    def __getitem__(self, idx):
        """
        Return data for the item at idx.

        Parameters
        ----------
        idx : sliceobj
            Indices to access data from

        Returns
        ----------
        item_dict : dict of np arrays
            Dict of the chosen items' data
            keys = "conditioning", "mic_audio", "mix_audio"
        """
        # load overall data
        item_dict = {}
        for key in self.dataset:
            item_dict[key] = self.dataset[key][idx]
        item_dict["conditioning"] = np.clip(item_dict["conditioning"], 0,127)
        return item_dict
