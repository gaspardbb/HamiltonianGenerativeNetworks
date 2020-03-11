from typing import Tuple, Callable

import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow import Tensor

from utils import _isin


class Hamiltonian:
    integrate_methods = ['leapfrog', 'euler']

    def __init__(self, dim, kinetic: Tuple[int] or Model = (128, 128), potential: Tuple[int] or Model = (128, 128)):
        """
        Separable Hamiltonian (kinetic and potential energies) and associated methods.

        Parameters
        ----------
        dim: Dimension of the input.
        kinetic, potential: Models for the energies. If a tuple is passed, will be a MLP with the associated number
        of layers and units.
        """
        self.kinetic = _hamiltonian_models(dim, kinetic, name="Kinetic")
        self.potential = _hamiltonian_models(dim, potential, name="Potential")
        self.dim = dim

    def __call__(self, q: Tensor, p: Tensor, *args, **kwargs) -> Tensor:
        """
        Returns value of the Hamiltonian in a given state.

        Parameters
        ----------
        q: value of the position, for the potential energy.
        p: value of the momentum, for the kinetic energy.
        """
        return self.potential(q) + self.kinetic(p)

    def grad_potential(self, q: Tensor) -> Tensor:
        with tf.GradientTape() as t:
            new_q = self.potential(q)
        return t.gradient(new_q, q)

    def integrate(self, q: Tensor, p: Tensor, n_steps: int, eps: float, forward, method="leapfrog"):
        _isin(method, Hamiltonian.integrate_methods)
        if method == "euler":
            # TODO: implement Euler integration
            raise NotImplementedError
        elif method == "leapfrog":
            return _leapfrog_integration(q=q, p=p, n_steps=n_steps, eps=eps, forward=forward,
                                         grad_func=self.grad_potential)


@tf.function
def _leapfrog_integration(q: Tensor, p: Tensor, n_steps: int, eps: float, forward: bool,
                          grad_func: Callable[[Tensor], Tensor]):
    # TODO: check this function!
    if not forward:
        eps = - eps

    # Make half step for the momentum
    p = p - eps/2 * grad_func(q)
    for i in range(n_steps):
        # Full step of position
        q = q + eps * p

        # Full step of momentum, except at the end
        if i != n_steps-1:
            p = p - eps * grad_func(q)

    # Make half step of momentum
    p = p - eps/2 * grad_func(q)
    return q, p


def _hamiltonian_models(dim, net: Tuple[int] or Model, name: str = ""):
    if isinstance(net, Model):
        assert net.input_shape == (dim,), ("Your custom net does not have the right shape!"
                                                  f"Got {net.input_shape}, expected {(dim,)}")
        return net

    return Sequential(([Dense(net[0], activation='softplus', input_shape=(dim,))]
                       + [Dense(n, activation='softplus') for n in net[1:]]
                       + [Dense(1, activation=None)]), name=name)

