from typing import Iterable, Tuple

import tensorflow as tf

from tensorflow import Tensor
from tensorflow_probability.python.distributions import Distribution
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.metrics import Mean

from encoder import GenericEncoder, GaussianEncoder
from hamiltonian import Hamiltonian


class NHF:

    def __init__(self, hamiltonians: Iterable[Hamiltonian], encoder: GenericEncoder, prior: Distribution,
                 n_steps: int = 20, eps: float = 1e-1, method="leapfrog"):
        for h in hamiltonians:
            assert encoder.dim == h.dim

        self.hamiltonians = hamiltonians
        self.encoder = encoder
        self.prior = prior
        self.dim = encoder.dim
        self.n_steps = n_steps
        self.eps = eps
        self.method = method

        self.n_epochs = 0
        self.compiled = False

        self.trainable_variables = []
        for h in self.hamiltonians:
            self.trainable_variables += h.trainable_variables
        self.trainable_variables += self.encoder.trainable_variables

        # To update easily the values of the dict. A NamedTuple class could be defined otherwise.
        self._log_mean = []
        self._prior_loss = []
        self._elbo = []
        self.history = {"log_mean": self._log_mean, "prior_loss": self._prior_loss, "elbo": self._elbo}

    def set_optimizer(self, optimizer="adam", lr=3e-4, additional_params={}):
        """
        Define an optimizer for the training loop.

        Parameters
        ----------
        optimizer: str or Optimizer
            The optimizer to choose.
        lr:
            The learning rate to use.
        additional_params:
            Dictionary of params passed to optimizers.get() if optimizer is not an Optimizer.

        Attributes
        ----------
        compiled
            True
        """
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            config = {"learning_rate": lr}
            config.update(additional_params)
            self.optimizer = tf.keras.optimizers.get({
                "class_name": "adam",
                "config": config}
            )
        self.compiled = True

    @tf.function
    def integrate(self, q: Tensor, p: Tensor, forward: bool):
        if forward:
            hamiltonians = self.hamiltonians
        else:
            hamiltonians = self.hamiltonians[::-1]

        for h in hamiltonians:
            q, p = h.integrate(q, p, forward=forward,
                               n_steps=self.n_steps, eps=self.eps, method=self.method)
        return q, p

    def prior_log_prob(self, q: Tensor, p: Tensor = None):
        """
        Computes the first term of the loss.

        Parameters
        ----------
        q, p:
            State. If the momentum is not provided, the value is evaluated from a single sample.

        Returns
        -------
        prior_log_proba
            The estimate of the first value of the loss.

        Notes
        -----
        This computes:
        E_{p ~ f(. | q)} [ln pi(H_1 ... H_T (q, p))]
        """
        if p is None:
            p = self.encoder.sample(q)

        q_T, p_T = q, p
        q_0, p_0 = self.integrate(q_T, p_T, forward=False)
        return self.prior.log_prob(q_0)

    # @tf.function
    def elbo(self, q: Tensor, p: Tensor = None, return_tuple=False):
        """
        Compute the ELBO for a given position. If the momentum is passed, then it is used as proxy for all the
        expectation computation. Otherwise, a momentum is sampled from the encoder, and will have the same shape as
        the position.

        Parameters
        ----------
        q, p:
            State. If p is not provided, it is sampled from the encoder.
        return_tuple:
            Whether to return the decomposition of the loss.

        Returns
        -------
        losses
            If return_tuple:
                prior_loss, log_mean, ELBO.
            Otherwise:
                ELBO
        """
        if p is None:
            p = self.encoder.sample(q)

        prior_loss = self.prior_log_prob(q, p)
        log_mean = self.encoder.logmean(q, p)

        if return_tuple:
            return prior_loss, log_mean, prior_loss - log_mean

        return prior_loss - log_mean

    def train(self, q_train: tf.data.Dataset, n_epochs: int):
        """
        Train all the networks.

        Parameters
        ----------
        q_train:
            Training position. Those are sampled from the density to approximate. Must be a tf.data.Dataset,
            for easy handling.
        n_epochs:
            Number of epochs on all the data.

        Notes
        -----
        The trainable variables are:
            * For each hamiltonian: the kinetic and potential nets
            * For the encoder: the mean and logvar nets
        All are accessible through {hamiltonian, encoder}.trainable_variables and are defined in the __init__ function.
        """
        if not self.compiled:
            raise ValueError(f"You need to set the optimizer before training the net!")

        sample = next(iter(q_train.take(1)))
        if sample.ndim == 1:
            # If there's only one dim, the dataset is not batched.
            q_train = q_train.batch(10)
        elif sample.ndim >= 3:
            # If there's 3 dims or more, there's obviously a problem.
            raise ValueError(f'Data should be 2 dim (batched 1 dim features). Got {sample.ndim}.'
                             f'Did you batched twice?')

        for epoch in range(n_epochs):
            epoch_log_mean = Mean()
            epoch_prior_loss = Mean()
            epoch_elbo = Mean()

            for batch in q_train:
                # We're looking at a single batch.
                # print(f"Doing new batch. Shape is: {batch.shape}")

                # Compute ELBO
                with tf.GradientTape() as tape:
                    prior_loss, log_mean, elbo = self.elbo(batch, return_tuple=True)

                gradient = tape.gradient(elbo, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))

                epoch_prior_loss(prior_loss)
                epoch_log_mean(log_mean)
                epoch_elbo(elbo)

            self.n_epochs += 1

    def sample(self, mc_samples: int, sample_shape: Tuple = (), **kwargs) -> tf.Tensor:
        """
        Perform a sample from the NHF.
        First, a state is sampled from the prior distribution. Then, it is integrated to produce a transformed state.

        Parameters
        ----------
        sample_shape
            Shape of the samples.
        n_steps, eps, method
            Parameters of the integration.

        Returns
        -------
        q
            A point of the target distribution.
        """
        raise NotImplementedError
        # TODO: explore all the p state?
        # Learn another encoder?
        # q_0 = self.prior.sample(sample_shape=sample_shape, **kwargs)
        # final_values = tf.zeros((mc_samples, self.dim))
        # for i in range(mc_samples):
        #     p = self.encoder.sample(q_0)
        #     final_values[i] = self.integrate(q_0, p, n_steps=self.n_steps, eps=self.eps, forward=True, method=method,
        #                                      **kwargs)
        # return tf.reduce_mean(final_values, axis=0)

    # @tf.function
    def evaluate(self, q: Tensor, n_samples: int = 10):
        """
        Evaluate the density at a given position q.
        The network is in fact evaluating the ELBO. So we take multiple samples, and keep only the best.

        Parameters
        ----------
        q:
            Position at which to evaluate the density.
        n_samples:
            Number of candidates for the density evaluation.

        Returns
        -------
        result
            Tensor of shape (batch_size,), containing the best ELBO for each position.
        """
        result = []
        for i in range(n_samples):
            result.append(self.elbo(q))
        result = tf.stack(result)
        return tf.reduce_max(result, axis=0)

    def grid_evaluation(self, x_range, y_range, granularity: int, n_samples: int = 1):
        assert self.dim == 2
        import numpy as np

        xx, yy = np.meshgrid(np.linspace(*x_range, granularity), np.linspace(*y_range, granularity))
        xy = np.stack([xx.flatten(), yy.flatten()]).T
        xy = tf.convert_to_tensor(xy, dtype="float32")

        proba_image = self.evaluate(xy, n_samples=n_samples).numpy().reshape((granularity, granularity))

        return proba_image


if __name__ == '__main__':
    from flow_example import FlowExample
    import tensorflow_probability as tfp

    # INPUT DATA
    # Define distribution
    gaussian = tfp.python.distributions.MultivariateNormalDiag(loc=tf.zeros(2), scale_identity_multiplier=1)
    distrib = FlowExample.from_tensorflow_distribution(gaussian)
    x = distrib.sample(20)
    # Cast to Dataset
    data = tf.data.Dataset.from_tensor_slices(x)
    data = data.batch(5)

    # INIT MODEL
    # Define hamiltonians
    h1_, h2_ = Hamiltonian(2), Hamiltonian(2)
    encoder_ = GaussianEncoder(2)
    prior_ = gaussian
    # Init
    nhf_ = NHF([h1_, h2_], encoder_, prior_)
    nhf_.set_optimizer()

    # TRAIN
    nhf_.train(data, 1)

    # SOME TEST
    print('Evaluating on zeros(4, 2) with 10 samples: \n %s'
          % nhf_.evaluate(tf.zeros((4, 2)), n_samples=10)
          )
    print('With 1 sample: \n %s'
          % nhf_.evaluate(tf.zeros((4, 2)), n_samples=1)
          )

    # GRID EVALUATION
    grid = nhf_.grid_evaluation((-1, 1), (-1, 1), granularity=10, n_samples=1)