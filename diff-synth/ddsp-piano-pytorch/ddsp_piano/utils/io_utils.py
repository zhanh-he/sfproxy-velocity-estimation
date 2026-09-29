import numpy as np
from pandas import read_csv

from ddsp_piano.utils.dataset import MaestroDataset


def dataset_from_csv(csv_path, split=None, device='cpu', cache_data_path=None, max_polyphony=16):
    """Load dataset from a csv file.
    Returns:
        - dataset (Pytorch Dataset): pytorch dataset from .csv
        - piano_models (list): list of different piano models in the dataset.
    """
    # .csv reading in pandas dataframe
    df = read_csv(csv_path)
    
    piano_models = np.sort(df['year'].unique())
    if split:
        df = df[df.split == split]
    df = df.to_dict('index')
    
    dataset = MaestroDataset(df, piano_models, max_polyphony=max_polyphony, split=split, cache_data_path=cache_data_path, device=device)
    
    return dataset


    
    
