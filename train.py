from typing import List

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from encoder import GaussianEncoder
from flow_example import FlowExample
from hamiltonian import Hamiltonian
from nhf import NHF


def standard_config():
    """Return a NHF with standard configuration."""
    h1, h2 = Hamiltonian(2), Hamiltonian(2)
    encoder = GaussianEncoder(2)
    prior = tfp.python.distributions.MultivariateNormalDiag(loc=tf.zeros(2), scale_identity_multiplier=1)
    nhf = NHF([h1, h2], encoder, prior)
    nhf.set_optimizer()
    return nhf


def train_gaussian(n_samples, n_epochs):
    nhf = standard_config()

    distribution = tfp.python.distributions.MultivariateNormalDiag(loc=tf.zeros(2), scale_identity_multiplier=1)
    distribution = FlowExample.from_tensorflow_distribution(distribution)
    x_train = distribution.sample(n_samples)

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.batch(10)

    nhf.train(dataset, n_epochs)

    plot_results(distribution, nhf, x_train)


def plot_results(distribution: FlowExample, nhf: NHF, samples: tf.Tensor, granularity=10):
    # Define the right scale
    samples = samples.numpy()
    x_min, y_min = np.nanmin(samples, axis=0)
    x_max, y_max = np.nanmax(samples, axis=0)
    x_range = x_min, x_max
    y_range = y_min, y_max

    axes: List[plt.Axes]
    fig, axes = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[1, 1, 2]))

    # Make first axes share the same scale
    axes[0].get_shared_x_axes().join(axes[0], axes[1])
    axes[0].get_shared_y_axes().join(axes[0], axes[1])

    axes[0].set_title("Original distribution")
    distribution.plot(x_range, y_range, granularity=granularity, show_samples=samples, ax=axes[0], show=False)

    axes[1].set_title("Predicted distribution")
    grid = nhf.grid_evaluation(x_range, y_range, granularity=granularity)
    axes[1].imshow(grid, extent=(*x_range, *y_range), cmap="coolwarm")

    axes[2].set_title("Losses")
    for k, v in nhf.history.items():
        numpy_values = [i.numpy() for i in v]
        axes[2].plot(numpy_values, label=k)
    plt.legend()

    plt.show()
    return axes

if __name__ == '__main__':
    train_gaussian(100, n_epochs=100)