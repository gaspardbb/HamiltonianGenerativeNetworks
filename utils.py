from typing import Iterable, Tuple

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense


def _isin(value, accepted_values: Iterable):
    if value not in accepted_values:
        raise ValueError(f"You passed {value}, but only the following values are accepted: {accepted_values}.")


def _mlp_models(dim, net: Tuple[int] or Model, activation: str, name: str = ""):
    if isinstance(net, Model):
        assert net.input_shape[1:] == (dim,), ("Your custom net does not have the right shape!"
                                               f"Got {net.input_shape}, expected {(dim,)}")
        return net

    return Sequential(([Dense(net[0], activation=activation, input_shape=(dim,))]
                       + [Dense(n, activation=activation) for n in net[1:]]
                       + [Dense(1, activation=None)]), name=name)