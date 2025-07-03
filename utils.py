import numpy as np
import tensorflow as tf

def normalization(data, parameters=None):
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:
        min_val = np.nanmin(norm_data, axis=0)
        max_val = np.nanmax(norm_data, axis=0)
        norm_data = (norm_data - min_val) / (max_val - min_val + 1e-6)

        norm_parameters = {'min_val': min_val, 'max_val': max_val}
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        norm_data = (norm_data - min_val) / (max_val - min_val + 1e-6)
        norm_parameters = parameters

    return norm_data, norm_parameters

def renormalization(norm_data, norm_parameters):
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']
    renorm_data = norm_data * (max_val - min_val + 1e-6) + min_val
    return renorm_data

def rounding(imputed_data, data_x):
    rounded_data = imputed_data.copy()
    for i in range(data_x.shape[1]):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
    return rounded_data

def rmse_loss(ori_data, imputed_data, data_m):
    ori_data, norm_params = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_params)

    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    return np.sqrt(nominator / (denominator + 1e-8))

def xavier_init(size):
    return tf.random.normal(shape=size, stddev=tf.math.sqrt(2.0 / sum(size)))

def binary_sampler(p, rows, cols):
    return np.random.binomial(1, p, size=(rows, cols))

def uniform_sampler(low, high, rows, cols):
    return np.random.uniform(low, high, size=(rows, cols))

def sample_batch_index(total, batch_size):
    return np.random.permutation(total)[:batch_size]


