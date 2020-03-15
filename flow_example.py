import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp

from tensorflow import Tensor
from tensorflow_probability.python.distributions import Distribution


class FlowExample:

    def __init__(self, dim):
        self.dim = dim

    def prob(self, q):
        raise NotImplementedError

    def sample(self, n_samples: int):
        raise NotImplementedError

    def plot(self, x_range, y_range, granularity=100, show_samples: Tensor=None, ax: plt.Axes=None, show=True):
        assert self.dim == 2
        xx, yy = np.meshgrid(np.linspace(*x_range, granularity),
                             np.linspace(*y_range, granularity))
        xy = np.stack([xx.flatten(), yy.flatten()]).T

        proba_image = self.prob(xy).numpy().reshape((granularity, granularity))

        if ax is None:
            ax: plt.Axes = plt.gca()

        ax.imshow(proba_image, extent=(*x_range, *y_range), cmap="coolwarm")

        if show_samples is not None:
            ax.scatter(show_samples[:, 0], show_samples[:, 1])

        if show:
            plt.show()

        return ax

    @classmethod
    def from_tensorflow_distribution(cls, distribution: Distribution or str):
        if type(distribution) is str:
            distribution = getattr(tfp.python.distributions, distribution)()

        distribution: Distribution
        shape = distribution.sample().shape.as_list()
        assert len(shape) == 1

        new_distrib = cls(dim=shape[0])
        new_distrib.prob = distribution.prob
        new_distrib.sample = distribution.sample

        return new_distrib


if __name__ == '__main__':
    gaussian = tfp.python.distributions.MultivariateNormalDiag(loc=np.zeros(2), scale_identity_multiplier=1)
    distrib = FlowExample.from_tensorflow_distribution(gaussian)
    x = distrib.sample(20)
    distrib.plot((-2, 2), (-2, 2), granularity=100, show_samples=x)

