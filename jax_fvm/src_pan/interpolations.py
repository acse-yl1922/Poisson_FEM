import jax.numpy as jnp
import grids

Array = grids.Array
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Grid = grids.Grid

# def InterpolateFromCellToBoundary():


def InterpolateGradientsFromCellToInteriorFace(
    foi: GridVariable,
    internal_c_grad_foi: Array,
    grid: Grid,
    method: str = "Linear",
) -> Array:

    internal_c_foi = foi.cell_phi
    p = grid.c_edge_index[0]
    q = grid.c_edge_index[1]
    gc = grid.ratio[:, 1]
    if method == "Linear":
        # this is pure linear interpolation
        internal_f_grad_foi = (
            gc[..., None] * internal_c_grad_foi[p]
            + (1 - gc)[..., None] * internal_c_grad_foi[q]
        )
    elif method == "Linear with Correction":
        # this is linear with correction:
        internal_f_grad_foi = (
            gc[..., None] * internal_c_grad_foi[p]
            + (1 - gc)[..., None] * internal_c_grad_foi[q]
        )
        internal_f_grad_foi += (
            -jnp.sum(internal_f_grad_foi * grid.eCF[: grid.N_if], axis=-1)[..., None]
            * grid.eCF[: grid.N_if]
            + (
                internal_c_foi[grid.c_edge_index[1]]
                - internal_c_foi[grid.c_edge_index[0]]
            )
            / grid.CF[: grid.N_if]
            * grid.eCF[: grid.N_if]
        )
    return internal_f_grad_foi
