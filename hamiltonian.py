from typing import Tuple, Callable

import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow import Tensor, Variable

from utils import _isin


class Hamiltonian:
    integrate_methods = ['leapfrog', 'euler']

    def __init__(self, dim, kinetic: Tuple[int] or Model = (128, 128), potential: Tuple[int] or Model = (128, 128)):
        """
        Separable Hamiltonian (kinetic and potential energies) and associated methods.

        Parameters
        ----------
        dim: Dimension of the input.
        kinetic, potential: Models for the energies. If a tuple istructuress passed, will be a MLP with the associated number
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

    # Really important decorator here! Otherwise, this is a python object. Then, when passed to leapfrog integration,
    # it will make retracing necessary at each step.
    @tf.function
    def grad_potential(self, q: Variable) -> Tensor:
        """Returns the gradient of the potential. Needed for integration."""
        with tf.GradientTape() as t:
            # Watch is necessary in case q is a Tensor (and not a Variable). Passing a Tensor without watch result in
            # None result.
            t.watch(q)
            potential = self.potential(q)
        return t.gradient(potential, q)

    @tf.function
    def grad_kinetic(self, p: Variable) -> Tensor:
        """Returns the gradient of the kinetic term. Needed for integration."""
        with tf.GradientTape() as t:
            t.watch(p)
            kinetic = self.kinetic(p)
        return t.gradient(kinetic, p)

    # @tf.function
    def integrate(self, q: Variable, p: Variable, n_steps: int, eps: float, forward, method="leapfrog"):
        """
        Perform integration of a state, according to Hamiltonian dynamics.

        Parameters
        ----------
        q, p:
            The current state (position, momentum).
        n_steps:
            The number of integration step to perform.
        eps:
            The size of each step.
        forward:
            Whether to perform a *forward* integration or a *backward* one, thanks to the reversibility of
            Hamiltonians dynamics.
        method: {"leapfrog", "euler"}
            Which numerical scheme to use. Leapfrog has eps**3 local error and eps**2 global error.
            Euler has eps**2, eps errors.

        Returns
        -------
        q, p:
            Tuple of new state: position, momentum.
        """
        _isin(method, Hamiltonian.integrate_methods)
        params = dict(q=q, p=p,
                      n_steps=tf.constant(n_steps),
                      eps=tf.constant(eps),
                      forward=tf.constant(forward),
                      grad_func_q=self.grad_potential,
                      grad_func_p=self.grad_kinetic)
        if method == "leapfrog":
            # Again, casting hyperparameters to tf.Tensor is crucial here. Otherwise, leapfrog will retrace at each
            # change of arguments.
            return _leapfrog_integration(**params)
        elif method == "euler":
            return _euler_integration(**params)


@tf.function
def _leapfrog_integration(q: Tensor, p: Tensor, n_steps: int, eps: float, forward: bool,
                          grad_func_q: Callable[[Tensor], Tensor],
                          grad_func_p: Callable[[Tensor], Tensor]):
    print(f"Tracing with {q}, {p}, {grad_func_q}, {grad_func_p}")
    if not forward:
        eps = - eps

    # Make half step for the momentum
    p = p - eps / 2 * grad_func_q(q)
    for i in range(n_steps):
        # Full step of position
        q = q + eps * grad_func_p(p)

        # Full step of momentum, except at the end
        if i != n_steps-1:
            p = p - eps * grad_func_q(q)

    # Make half step of momentum
    p = p - eps / 2 * grad_func_q(q)
    return q, p


@tf.function
def _euler_integration(q: Tensor, p: Tensor, n_steps: int, eps: float, forward: bool,
                          grad_func_q: Callable[[Tensor], Tensor],
                          grad_func_p: Callable[[Tensor], Tensor]):
    print(f"Tracing with {q}, {p}, {grad_func_q}, {grad_func_p}")
    if not forward:
        eps = - eps

    for i in range(n_steps):
        q = q + eps * grad_func_p(p)

        p = p - eps * grad_func_q(q)

    return q, p


def _hamiltonian_models(dim, net: Tuple[int] or Model, name: str = ""):
    if isinstance(net, Model):
        assert net.input_shape[1:] == (dim,), ("Your custom net does not have the right shape!"
                                               f"Got {net.input_shape}, expected {(dim,)}")
        return net

    return Sequential(([Dense(net[0], activation='softplus', input_shape=(dim,))]
                       + [Dense(n, activation='softplus') for n in net[1:]]
                       + [Dense(1, activation=None)]), name=name)


def test_integration(method='leapfrog'):
    """
    Test integration on a simple example:
    U(q) = q^2/2
    K(p) = p^2/2
    3 plots, side by side. Forward motion (in blue) then backward motion (in red).
    """
    n_steps = 20
    list_eps = [0.3, 0.8, 1.5]

    initial_p = tf.ones(1)
    initial_q = tf.zeros(1)

    simulations = np.zeros((3, 2*n_steps, 2))
    energies = np.zeros((3, 2*n_steps))

    def square_layer():
        x = Input(shape=(1, ))
        return Model(inputs=x, outputs=tf.square(x) / 2)

    hamiltonian = Hamiltonian(1, kinetic=square_layer(), potential=square_layer())

    for r, e, eps in zip(simulations, energies, list_eps):
        q, p = initial_q, initial_p
        for i in range(n_steps):
            q, p = hamiltonian.integrate(q, p, n_steps=1,
                                  eps=eps, forward=True, method=method)
            r[i] = q.numpy(), p.numpy()
            e[i] = hamiltonian(q, p)
        for i in range(n_steps):
            q, p = hamiltonian.integrate(q, p, n_steps=1, eps=eps, forward=False, method=method)
            r[n_steps+i] = q.numpy(), p.numpy()
            e[n_steps+i] = hamiltonian(q, p)

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    fig, axes = plt.subplots(2, 3)

    baseline = np.linspace(0, 1, 1000)
    baseline_x = np.cos(baseline * 2 * np.pi)
    baseline_y = np.sin(baseline * 2 * np.pi)

    ax: plt.Axes
    fig: plt.Figure
    for r, eps, ax in zip(simulations, list_eps, axes[0]):
        ax.plot(r[:n_steps, 0], r[:n_steps, 1], 'b+-')
        ax.plot(r[n_steps:, 0], r[n_steps:, 1], 'r+-')
        ax.plot(baseline_x, baseline_y, 'k')
        ax.set_aspect('equal')
        ax.set_title(f'$\epsilon = {eps}$')
    for e, ax in zip(energies, axes[1]):
        ax.plot(e)
        ax.axhline(e[0], linestyle="--", color="k")
        ax.axvline(n_steps, linestyle="-.", color="k")
    fig.suptitle("$n_{steps} = %d$" % n_steps)
    plt.show()

    return simulations


if __name__ == '__main__':
    result = test_integration(method="leapfrog")