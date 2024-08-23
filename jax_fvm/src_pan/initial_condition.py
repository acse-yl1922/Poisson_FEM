import grids
import numpy as np
import jax.numpy as jnp
from typing import Any, Callable, Optional, Sequence, Tuple, Union

Array = grids.Array
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


def InitialScalarField(
    initial_info: Any, grid: grids.Grid, scalar_bc: grids.BoundaryConditions, **kwarg
) -> GridVariable:
    """Given scalar functions on arrays, returns the velocity field on the grid."""
    if initial_info == float or (initial_info == Array and initial_info.shape == (1,)):
        cell_phi = jnp.ones(grid.N_c)
        bd_phi = jnp.ones(grid.N_bdf)
    elif type(initial_info) == Array:
        cell_phi = initial_info
        bd_phi = cell_phi[grid.face_owner[grid.N_if :]]
    elif callable(initial_info):
        cell_phi = grid.EvaluateOnMesh(initial_info, "cell")
        bd_phi = grid.EvaluateOnMesh(initial_info, "boundary")
    else:
        raise Exception("input type is not supported")
    return GridVariable(cell_phi, bd_phi, scalar_bc, **kwarg)
