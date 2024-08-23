import grids
import numpy as np
import jax.numpy as jnp
from jax import lax


Array = grids.Array
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Grid = grids.Grid


def AssembleMatrixForm(
    FluxCf:Array, FluxFf:Array, FluxVf:Array, FluxCb:Array, FluxFb:Array, FluxVb:Array, foi: GridVariable, grid: Grid
):
    """
    Assembles the matrix form of a discretized equation system for a finite volume method, 
    returns the diagonal and off-diagonal entries of the matrix as well as the right-hand side vector.

    Parameters
    ----------
    FluxCf, FluxFf, FluxVf : Array
        Linear and nonlinear fluxes at internal faces.
    FluxCb, FluxFb, FluxVb : Array
        Linear and nonlinear fluxes at boundary faces.
    
    FluxC, FluxC_old, FluxV
        Coefficient for the current/previous time step and additional transient flux term.
    foi
        The grid variable object containing the scalar field `cell_phi` and boundary field `bd_phi`.
    grid
        The grid object containing the grid structure

    Returns
    -------
    a_C : ndarray
        The diagonal (main) coefficients of the matrix, corresponding to each cell's contribution.
    a_nb : tuple
        A tuple containing the indices of the neighbor cells and their associated off-diagonal 
        coefficients (neighbor contributions).
    b_C : ndarray
        The right-hand side vector
    """

    # we first added up those values for a_C and b_C
    a_C = lax.scatter_add(
        jnp.zeros((grid.N_c)),
        grid.face_owner_ud[..., None],
        jnp.concatenate((FluxCf, -FluxFf, FluxCb)),  # <N_f, 1> * <N_f, 3>
        grid.scalar_scatter_dim_num,
    )

    a_nb = (
        jnp.hstack((grid.c_edge_index, jnp.flip(grid.c_edge_index, axis=0))),
        jnp.concatenate((FluxFf, -FluxCf)),
    )

    b_C = grid.source * grid.c_vol
    # b_C = grid.source*grid.c_vol*0.
    # print('debug b_C_source', b_C[:10])
    b_C -= lax.scatter_add(
        jnp.zeros((grid.N_c)),
        grid.face_owner_ud[..., None],
        jnp.concatenate(
            (
                FluxCf * foi.cell_phi[grid.c_edge_index[0]]
                + FluxFf * foi.cell_phi[grid.c_edge_index[1]]
                + FluxVf,
                -FluxFf * foi.cell_phi[grid.c_edge_index[1]]
                - FluxCf * foi.cell_phi[grid.c_edge_index[0]]
                - FluxVf,
                FluxCb * foi.cell_phi[grid.face_owner[grid.N_if :]]
                + FluxFb * foi.bd_phi
                + FluxVb,
            )
        ),  # <N_f, 1> * <N_f, 3>
        grid.scalar_scatter_dim_num,
    )  # this is the equation residual

    return a_C, a_nb, b_C

def Assemble_matrix_transient(FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb, FluxC,FluxC_old, FluxV,  foi: GridVariable, grid: Grid):
    """
    Assembles the matrix form of a discretized equation system with a transient term.

    Parameters
    ----------
    FluxCf, FluxFf, FluxVf : Array
        Linear and nonlinear fluxes at internal faces.
    FluxCb, FluxFb, FluxVb : Array
        Linear and nonlinear fluxes at boundary faces.
    foi
        The grid variable object containing the scalar field `cell_phi` and boundary field `bd_phi`.
    grid
        The grid object containing the grid structure

    Returns
    -------
    a_C : ndarray
        The diagonal (main) coefficients of the matrix, corresponding to each cell's contribution.
    a_nb : tuple
        A tuple containing the indices of the neighbor cells and their associated off-diagonal 
        coefficients (neighbor contributions).
    b_C : ndarray
        The right-hand side vector
    """
    # we first added up those values for a_C and b_C
    a_C = lax.scatter_add(
        jnp.zeros((grid.N_c)),
        grid.face_owner_ud[..., None],
        jnp.concatenate((FluxCf, -FluxFf, FluxCb)),  # <N_f, 1> * <N_f, 3>
        grid.scalar_scatter_dim_num,
    )

    a_nb = (
        jnp.hstack((grid.c_edge_index, jnp.flip(grid.c_edge_index, axis=0))),
        jnp.concatenate((FluxFf, -FluxCf)),
    )

    b_C = grid.source * grid.c_vol
    # b_C = grid.source*grid.c_vol*0.
    # print('debug b_C_source', b_C[:10])
    b_C -= lax.scatter_add(
        jnp.zeros((grid.N_c)),
        grid.face_owner_ud[..., None],
        jnp.concatenate(
            (
                FluxCf * foi.cell_phi[grid.c_edge_index[0]]
                + FluxFf * foi.cell_phi[grid.c_edge_index[1]]
                + FluxVf,
                -FluxFf * foi.cell_phi[grid.c_edge_index[1]]
                - FluxCf * foi.cell_phi[grid.c_edge_index[0]]
                - FluxVf,
                FluxCb * foi.cell_phi[grid.face_owner[grid.N_if :]]
                + FluxFb * foi.bd_phi
                + FluxVb,
            )
        ),  # <N_f, 1> * <N_f, 3>
        grid.scalar_scatter_dim_num,
    )  # this is the equation residual
    
    # Adding trasient Flux coefficient
    a_C += FluxC
    b_C = b_C - FluxV - foi.cell_phi*FluxC_old
    
    return a_C, a_nb, b_C
