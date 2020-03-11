from typing import Iterable


def _isin(value, accepted_values: Iterable):
    if value not in accepted_values:
        raise ValueError(f"You passed {value}, but only the following values are accepted: {accepted_values}.")