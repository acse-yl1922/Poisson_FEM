from advection_diffusion import *
from assemble import *

import jax
from functools import partial
from typing import Callable

def solve_cg(a_C, a_nb, b_C, x0, grid, tol, atol, maxiter):
    """
    solves the system Ax=b_C using a_C, a_nb, b_C

    Inputs:
    -------
        a_C:
            diagonal component of A

        a_nb:
            off-diagonal component of A, specified in csr_matrix. (<row, col>, value)

        b_C:
            constants at RHS
    """

    def linear_op(x):
        x_new = x * a_C
        x_new += lax.scatter_add(
            jnp.zeros((grid.N_c)),
            a_nb[0][0][..., None],
            x[a_nb[0][1]] * a_nb[1],
            grid.scalar_scatter_dim_num,
        )  # this is the equation residual
        return x_new

    x, _ = jax.scipy.sparse.linalg.cg(
        linear_op, b_C, x0=x0, tol=tol, atol=atol, maxiter=maxiter
    )
    residual = b_C - linear_op(x)

    return x, residual

@partial(jax.jit, static_argnums=(1,))
def SolveSteady_jit(
    foi: GridVariable,
    grid: Grid,
):

    FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb = DiffusionFlux_linear(
        foi, grid
    )
    a_C, a_nb, b_C = AssembleMatrixForm(
        FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb, foi, grid
    )
    x0 = jnp.zeros_like(foi.cell_phi)
    tol = 1e-8
    atol = 1e-8
    max_iter = 1000
    dphi, residual = solve_cg(a_C, a_nb, b_C, x0, grid, tol, atol, max_iter)
    foi_new = GridVariable(
        dphi + foi.cell_phi, foi.bd_phi, foi.bc, foi.name
    ).UpdateBoundaryPhi(grid)
    # return foi_new,b_C, residual, dphi # only for debugging
    return foi_new, residual


def SolveSteady(
    foi: GridVariable,
    grid: Grid,
):

    FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb = DiffusionFlux_linear(
        foi, grid
    )
    a_C, a_nb, b_C = AssembleMatrixForm(
        FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb, foi, grid
    )
    x0 = jnp.zeros_like(foi.cell_phi)
    tol = 1e-8
    atol = 1e-8
    max_iter = 1000
    dphi, residual = solve_cg(a_C, a_nb, b_C, x0, grid, tol, atol, max_iter)
    foi_new = GridVariable(
        dphi + foi.cell_phi, foi.bd_phi, foi.bc, foi.name
    ).UpdateBoundaryPhi(grid)
    # return foi_new,b_C, residual, dphi # only for debugging
    return foi_new, residual

@partial(jax.jit, static_argnums=(1,))
def Transient_Downwind_jit(
    foi: GridVariable,
    grid: Grid,
):
    phi_old = foi.cell_phi
    
    FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb = DiffusionFlux_linear(
    foi, grid) #diffusion flux
    FluxC, FluxC_old, FluxV = TransientFlux(grid)# Transient flux
    a_C, a_nb, b_C = Assemble_matrix_transient(FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb,FluxC, FluxC_old, FluxV,foi,grid)
    
    foi = GridVariable(phi_old+grid.deltaT*(a_C+a_nb)*phi_old,foi.bd_phi, foi.bc, foi.name).UpdateBoundaryPhi(grid)
    return foi

def Transient_Downwind(
    foi: GridVariable,
    grid: Grid,
):
    phi_old = foi.cell_phi
    RHS, residual = SolveSteady(foi, grid)
    foi = GridVariable(phi_old+grid.deltaT*RHS.cell_phi/grid.rho,foi.bd_phi, foi.bc, foi.name).UpdateBoundaryPhi(grid)
    return foi, residual


@partial(jax.jit, static_argnums=(1,))
def Transient_upwind_jit(
    foi: GridVariable,
    grid: Grid,
):

    FluxC, FluxC_old, FluxV = TransientFlux(grid)
    FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb = DiffusionFlux_linear(foi, grid)
    a_C, a_nb, b_C = Assemble_matrix_transient(FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb,FluxC, FluxC_old, FluxV,foi,grid)
    x0 = jnp.zeros_like(foi.cell_phi)
    tol = 1e-8
    atol = 1e-8
    max_iter = 1000
    dphi, residual = solve_cg(a_C, a_nb, b_C, x0, grid, tol, atol, max_iter)
    foi_new = GridVariable(
        dphi + foi.cell_phi, foi.bd_phi, foi.bc, foi.name
    ).UpdateBoundaryPhi(grid)
    return foi_new, residual


def Transient_upwind(
    foi: GridVariable,
    grid: Grid,
):
    FluxC, FluxC_old, FluxV = TransientFlux(grid)
    FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb = DiffusionFlux_linear(foi, grid)
    a_C, a_nb, b_C = Assemble_matrix_transient(FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb,FluxC, FluxC_old, FluxV,foi,grid)
    x0 = jnp.zeros_like(foi.cell_phi)
    tol = 1e-8
    atol = 1e-8
    max_iter = 1000
    dphi, residual = solve_cg(a_C, a_nb, b_C, x0, grid, tol, atol, max_iter)
    foi_new = GridVariable(
        dphi + foi.cell_phi, foi.bd_phi, foi.bc, foi.name
    ).UpdateBoundaryPhi(grid)
    return foi_new, residual