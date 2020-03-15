from typing import Tuple

import tensorflow as tf

from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow_probability.python.distributions import MultivariateNormalDiag

from utils import _mlp_models


class GenericEncoder:

    dim = None

    def sample(self, q: Tensor, *args, **kwargs):
        raise NotImplementedError

    def logmean(self, q: Tensor, p: Tensor = None):
        raise NotImplementedError

    @property
    def trainable_variables(self):
        raise NotImplementedError


class GaussianEncoder(GenericEncoder):

    def __init__(self, dim: int, mean: Tuple[int] or Model = (128, 128), var: Tuple[int] or Model = (128, 128)):
        self.mean_net = _mlp_models(dim, mean, activation="relu", name="Mean")
        self.logvar_net = _mlp_models(dim, var, activation="relu", name="Var")
        self.dim = dim
        self._gaussian_sampler = MultivariateNormalDiag(loc=tf.zeros(dim), scale_identity_multiplier=1)

    def __call__(self, q, *args, **kwargs):
        return self.mean_net(q), self.logvar_net(q)

    def sample(self, q: Tensor, *args, **kwargs):
        batch_size = q.shape[0]
        epsilon = self._gaussian_sampler.sample(batch_size)
        # That way, epsilon has same shape than q
        mu, sigma = self(q)
        return mu + epsilon * tf.exp(0.5 * sigma)

    def logmean(self, q: Tensor, p: Tensor = None):
        """
        Returns the expectancy of log(f(p | q) when p has law f(. | q).
        An closed-form formula may exists. But as the other term in the loss is computed with SGD, we can do the same
        here.

        Parameters
        ----------
        q, p:
            state. If the momentum is not provided, it is sampled.

        Returns
        -------
        A real number, value of the expectancy. Without 2 pi.
        """
        if p is None:
            p = self.sample(q)
        mu, sigma = self(q)
        log_det = tf.reduce_sum(sigma)
        exp_term = 1/2 * tf.reduce_sum(tf.square(p - mu) * tf.exp(- sigma))
        return - log_det - exp_term

    @property
    def trainable_variables(self):
        return self.mean_net.trainable_variables + self.logvar_net.trainable_variables