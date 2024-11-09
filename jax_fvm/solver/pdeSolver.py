from base import grids
from base.diffusion import DiffusionFlux_linear
from base.assemble import AssembleMatrixForm_steady_state, forward_euler,implicitEuler, backward, crank_nicolson
from base.timestepping import FirstOrderEulerTransientTerm, CK, SOUE

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from solver.cgSolve import cg

# The scipy package import is for experimental PCG
# import numpy as np
# from scipy.sparse.linalg import cg,spilu,LinearOperator
# from scipy.sparse import coo_matrix

from typing import Callable

Array = grids.Array
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Grid = grids.Grid

def solve_cg(a_C, a_nb, b_C, x0, grid, tol, atol, maxiter):
    """
    solves the system Ax=b_C using a_C, a_nb, b_C

    Inputs:
    -------
        a_C:
            diagonal component of A

        a_nb:
            off-diagonal component of A

        b_C:
            constants at RHS
    """

    def linear_op(x):
        x_new = x * a_C
        x_new += lax.scatter_add(
            jnp.zeros((grid.n_cells)),
            a_nb[0][0][..., None],
            x[a_nb[0][1]] * a_nb[1],
            grid.scalar_scatter_dim_num,
        )
        return x_new
    # The following code is used for pre-implementation of PCG solver
    # if grid.preconditioner == 'none':
    #     M = None
    
    # if grid.preconditioner == 'ILU':
    #     try:
    #         a_nb_indices = np.array(a_nb[0]).T
    #         a_nb_value = np.array(a_nb[1])
    #         coords_diag = np.array([[i, i] for i in range(grid.n_cells)])
    #         coords = np.concatenate([coords_diag, a_nb_indices])
    #         data = np.concatenate([a_C, a_nb_value])
    #         row, col = coords[:, 0], coords[:, 1]
    #         A_sparse = coo_matrix((data, (row, col)), shape=(grid.n_cells, grid.n_cells))
    #         ilu = spilu(A_sparse)
    #         M_x = lambda x: ilu.solve(x)
    #         M = LinearOperator(shape=A_sparse.shape, matvec=M_x)
    #     except Exception as e:
    
    #         raise RuntimeError(f"ILU failed: {e}. Check if matrix is SPD")
    # elif grid.preconditioner == 'DIC':
    #     # # Compute DIC preconditioner
    #     diag_A = np.array(a_C)
    #     # Ensure diagonal entries are positive
    #     diag_A[diag_A <= 0] = 1e-10  # Small positive number to prevent division by zero or negative sqrt
    #     M_diag = 1.0 / np.sqrt(diag_A)
    #     # Create a linear operator for the preconditioner
    #     def M_x(x):
    #         return M_diag * x
    #     M = LinearOperator(shape=A_sparse.shape, matvec=M_x)
        
    x, info = cg(linear_op, b_C, x0=x0, tol=tol, atol=atol, maxiter=maxiter) 
    return x, info


@partial(jax.jit, static_argnums=(1, 2))
def LaplacianJax_jit(foi: GridVariable, grid: Grid, source_function: Callable):
    FluxCf, FluxFf, FluxVf, FluxCb, FluxFb = DiffusionFlux_linear(foi, grid)
    if grid.ddtSchemes == "steadyState":
        a_C, a_nb, b_C = AssembleMatrixForm_steady_state(
            FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, foi, grid, source_function
        )
    elif grid.ddtSchemes == "Euler":
        FluxC, FluxC_old, _ = FirstOrderEulerTransientTerm(grid)
        a_C, a_nb, b_C = implicitEuler(
            FluxCf,
            FluxFf,
            FluxVf,
            FluxCb,
            FluxFb,
            FluxC,
            FluxC_old,
            foi,
            grid,
            source_function,
        )
    elif grid.ddtSchemes == "forwardEuler":
        return forward_euler(FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, foi, grid,source_function)

    x0 = jnp.zeros_like(foi.cell_phi)
    tol = 1e-08
    atol = 1e-08
    max_iter = 1000
    dphi, info = solve_cg(a_C, a_nb, b_C, x0, grid, tol, atol, max_iter)
    new_timestep = foi.timestep + grid.deltaT
    foi_new = GridVariable(
        dphi + foi.cell_phi,
        foi.bd_phi,
        new_timestep,
        foi.bc,
        foi.name,
    ).UpdateBoundaryPhi(grid)
    info["phi"] = dphi + foi.cell_phi
    return foi_new, info

@partial(jax.jit, static_argnums=(1,2))
def LaplacianJax_high_order_jit(fois, grid: Grid, source_function: Callable):
    foi_old, foi = fois
    FluxCf, FluxFf, FluxVf, FluxCb, FluxFb = DiffusionFlux_linear(foi, grid)
    _, _, FluxVf_old, _, _ = DiffusionFlux_linear(foi_old, grid)
    
    if grid.ddtSchemes == "CrankNicolson 1":
        FluxC, FluxC_old, _ = CK(grid)
        a_C, a_nb, b_C = crank_nicolson(
            FluxCf,
            FluxFf,
            FluxVf,
            FluxCb,
            FluxFb,
            FluxC,
            FluxC_old,
            FluxVf_old,
            foi,
            grid,
            source_function,
        )
    elif grid.ddtSchemes == "backward":
        # return NotImplementedError('Function under development')
        FluxC, FluxC_old, FluxC_old_old = SOUE(grid)
        a_C, a_nb, b_C = backward(
            FluxCf,
            FluxFf,
            FluxVf,
            FluxCb,
            FluxFb,
            FluxC,
            FluxC_old,
            FluxC_old_old,
            foi,
            foi_old,
            grid,
            source_function,
        )

    x0 = jnp.zeros_like(foi.cell_phi)
    tol = 1e-08
    atol = 1e-08
    max_iter = 1000
    dphi, info = solve_cg(a_C, a_nb, b_C, x0, grid, tol, atol, max_iter)
    new_timestep = foi.timestep + grid.deltaT
    foi_new = GridVariable(
        dphi + foi.cell_phi,
        foi.bd_phi,
        new_timestep,
        foi.bc,
        foi.name,
    ).UpdateBoundaryPhi(grid)
    info["phi"] = dphi + foi.cell_phi
    return [foi, foi_new], info