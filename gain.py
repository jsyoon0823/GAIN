import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import binary_sampler, uniform_sampler, sample_batch_index


class Generator(tf.keras.Model):
    def __init__(self, dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(dim, activation='relu')
        self.output_layer = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


class Discriminator(tf.keras.Model):
    def __init__(self, dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(dim, activation='relu')
        self.output_layer = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


def gain(data_x, gain_parameters):
    '''Impute missing values in data_x using GAIN.'''

    data_m = 1 - np.isnan(data_x)
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    no, dim = data_x.shape

    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, nan=0)

    G = Generator(dim)
    D = Discriminator(dim)

    g_optimizer = tf.keras.optimizers.Adam()
    d_optimizer = tf.keras.optimizers.Adam()

    for it in tqdm(range(iterations)):
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :].astype(np.float32)
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim).astype(np.float32)
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim).astype(np.float32)
        H_mb = M_mb * H_mb_temp


        X_mb_observed = M_mb * X_mb + (1 - M_mb) * Z_mb


        with tf.GradientTape() as d_tape:
            G_sample = G(tf.concat([X_mb_observed, M_mb], axis=1))
            Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
            D_prob = D(tf.concat([Hat_X, H_mb], axis=1))
            D_loss = -tf.reduce_mean(M_mb * tf.math.log(D_prob + 1e-8) +
                                      (1 - M_mb) * tf.math.log(1. - D_prob + 1e-8))

        d_gradients = d_tape.gradient(D_loss, D.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, D.trainable_variables))

        with tf.GradientTape() as g_tape:
            G_sample = G(tf.concat([X_mb_observed, M_mb], axis=1))
            Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
            D_prob = D(tf.concat([Hat_X, H_mb], axis=1))

            G_loss_temp = -tf.reduce_mean((1 - M_mb) * tf.math.log(D_prob + 1e-8))
            MSE_loss = tf.reduce_mean((M_mb * X_mb - M_mb * G_sample) ** 2) / tf.reduce_mean(M_mb)
            G_loss = G_loss_temp + alpha * MSE_loss

        g_gradients = g_tape.gradient(G_loss, G.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, G.trainable_variables))

    Z_mb = uniform_sampler(0, 0.01, no, dim)
    X_mb_observed = data_m * norm_data_x + (1 - data_m) * Z_mb
    imputed_data = G(tf.concat([X_mb_observed, data_m], axis=1)).numpy()
    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    imputed_data = renormalization(imputed_data, norm_parameters)
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data

