# coding=utf-8
'''Data loader for UCI letter, spam, and MNIST datasets.'''

import numpy as np
from utils import binary_sampler
from keras.datasets import mnist
import os

def data_loader(data_name, miss_rate):
    '''
    Loads datasets and introduces missingness.

    Args:
        - data_name: 'letter', 'spam', or 'mnist'
        - miss_rate: the probability of missing components (float between 0 and 1)

    Returns:
        - data_x: Original data
        - miss_data_x: Data with missing values
        - data_m: Mask matrix (1 = observed, 0 = missing)
    '''

    # Load data
    if data_name.lower() in ['letter', 'spam']:
        file_path = os.path.join('data', f'{data_name}.csv')
        try:
            data_x = np.loadtxt(file_path, delimiter=",", skiprows=1)
        except Exception as e:
            raise FileNotFoundError(f"Unable to load {file_path}: {str(e)}")
    elif data_name.lower() == 'mnist':
        (data_x, _), _ = mnist.load_data()
        data_x = np.reshape(data_x, [data_x.shape[0], -1]).astype(np.float32)
    else:
        raise ValueError("data_name must be one of ['letter', 'spam', 'mnist']")

    # Get shape
    no, dim = data_x.shape

    # Introduce missing values
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m

