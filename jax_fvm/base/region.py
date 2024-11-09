from __future__ import annotations

import dataclasses
import numbers
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
from jax import core, vmap, lax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np
import os

# TODO(jamieas): consider moving common types to a separate module.
# TODO(shoyer): consider adding jnp.ndarray?
Array = Union[np.ndarray, jax.Array]
IntOrSequence = Union[int, Sequence[int]]

# There is currently no good way to indicate a jax "pytree" with arrays at its
# leaves. See https://jax.readthedocs.io/en/latest/jax.tree_util.html for more
# information about PyTrees and https://github.com/google/jax/issues/3340 for
# discussion of this issue.
PyTree = Any


@dataclasses.dataclass(init=False, frozen=False)
class Region:
    """case class that stores everything"""

    def __init__(
        self,
    ):
        self.foamDictionary = dict()
        self.fluid = dict()
