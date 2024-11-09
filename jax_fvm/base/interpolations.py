from base import grids

Array = grids.Array 
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Grid = grids.Grid


def scalar_linear_interpolation(
    scalar: Array,
    grid: Grid,
) -> Array:

    p = grid.edge_index[0]
    q = grid.edge_index[1]
    gc = 1 - grid.ratio[: grid.n_I_faces]

    # this is pure linear interpolation
    internal_face_scalar = gc * scalar[p] + (1 - gc) * scalar[q]  # cell quantity

    return internal_face_scalar


def vector_linear_interpolation(
    vector: Array,
    grid: Grid,
) -> Array:

    p = grid.edge_index[0]
    q = grid.edge_index[1]
    gc = 1 - grid.ratio[: grid.n_I_faces]

    # this is pure linear interpolation
    internal_face_grad_phi = gc[..., None] * vector[p] + (1 - gc)[..., None] * vector[q]

    return internal_face_grad_phi
