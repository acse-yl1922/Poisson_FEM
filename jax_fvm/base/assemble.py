from base import grids
import jax.numpy as jnp
from jax import lax
from typing import Callable
from base.timestepping import FirstOrderEulerTransientTerm,ButcherTableau

Array = grids.Array
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Grid = grids.Grid

def AssembleMatrixForm_steady_state(
    FluxCf,
    FluxFf,
    FluxVf,
    FluxCb,
    FluxFb,
    foi: GridVariable,
    grid: Grid,
    source_function: Callable,
):
    # we first added up those values for a_C and b_C
    a_C = lax.scatter_add(
        jnp.zeros((grid.n_cells)),
        grid.face_owner_ud[..., None],
        jnp.concatenate((FluxCf, -FluxFf, FluxCb)),  # <N_f, 1> * <N_f, 3>
        grid.scalar_scatter_dim_num,
    )

    a_nb = (
        jnp.hstack((grid.edge_index, jnp.flip(grid.edge_index, axis=0))),
        jnp.concatenate((FluxFf, -FluxCf)),
    )
    source = source_function(grid.cell_centers)
    b_C = source * grid.cell_volumes
    if grid.snGradSchemes == "uncorrected" or grid.snGradSchemes == "orthogonal":
        b_C -= lax.scatter_add(
            jnp.zeros((grid.n_cells)),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]],
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]],
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )  # this is the equation residual

    elif grid.snGradSchemes == "corrected":
        b_C -= lax.scatter_add(
            jnp.zeros((grid.n_cells)),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]]
                    + FluxVf,
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]]
                    - FluxVf,
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                    # + FluxVb,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )  # this is the equation residual

    return a_C, a_nb, b_C


def implicitEuler(
    FluxCf,
    FluxFf,
    FluxVf,
    FluxCb,
    FluxFb,
    FluxC,
    FluxC_old,
    foi: GridVariable,
    grid: Grid,
    source_function: Callable,
):

    a_C = lax.scatter_add(
        FluxC,
        grid.face_owner_ud[..., None],
        jnp.concatenate((FluxCf, -FluxFf, FluxCb)),  # <N_f, 1> * <N_f, 3>
        grid.scalar_scatter_dim_num,
    )

    a_nb = (
        jnp.hstack((grid.edge_index, jnp.flip(grid.edge_index, axis=0))),
        jnp.concatenate((FluxFf, -FluxCf)),
    )

    source = source_function(grid.cell_centers, foi.timestep + grid.deltaT)
    b_C = source * grid.cell_volumes
    b_C = b_C - foi.cell_phi * FluxC_old + -FluxC * foi.cell_phi

    if grid.snGradSchemes == "orthogonal" or grid.snGradSchemes == "uncorrected":
        b_C -= lax.scatter_add(
            jnp.zeros((grid.n_cells)),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]],
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]],
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )  # this is the equation residual

    elif grid.snGradSchemes == "corrected":
        b_C -= lax.scatter_add(
            jnp.zeros((grid.n_cells)),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]]
                    + FluxVf,
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]]
                    - FluxVf,
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                    # + FluxVb,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )  # this is the equation residual
    return a_C, a_nb, b_C


def forward_euler(
    FluxCf,
    FluxFf,
    FluxVf,
    FluxCb,
    FluxFb,
    foi: GridVariable,
    grid: Grid,
    source_function: Callable,
):

    FluxC, FluxC_old, _ = FirstOrderEulerTransientTerm(grid)
    source = source_function(grid.cell_centers, foi.timestep)
    b_C = source * grid.cell_volumes - foi.cell_phi * FluxC_old

    if grid.snGradSchemes == "orthogonal" or grid.snGradSchemes == "uncorrected":
        b_C -= lax.scatter_add(
            jnp.zeros((grid.n_cells)),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]]
                    ,
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]]
                    ,
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                    # + FluxVb,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )  # this is the equation residual
    
    if grid.snGradSchemes == "corrected":
        # raise NotImplementedError('Crank Nicolson + gauss linear corrected has issue now, fixing code bug')
        b_C -= lax.scatter_add(
            jnp.zeros(grid.n_cells),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]]
                    + FluxVf,
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]]
                    - FluxVf,
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )
    new_timestep = foi.timestep + grid.deltaT
    foi_new = GridVariable(b_C / FluxC, foi.bd_phi, new_timestep, foi.bc, foi.name).UpdateBoundaryPhi(
        grid
    )
    return foi_new, b_C / FluxC

def runge_kutta_time_stepping(
    grid:Grid,
    foi:GridVariable,
    source_function: Callable,
    FluxCf, FluxFf, FluxVf, FluxCb, FluxFb,
):
    # Butcher tableau for the given method
    tableau = ButcherTableau(grid.ddtSchemes)

    # Get the coefficients from the tableau
    a, b, c = tableau.a, tableau.b, tableau.c
    # Number of stages
    nstage = len(b)
    y = foi.cell_phi
    
    derivatives = [jnp.zeros(grid.n_cells) for _ in range(nstage)]

    # Calculate the intermediate stages
    for i in range(nstage):
        k = foi
        b_C = source_function(grid.cell_centers, foi.timestep + grid.deltaT * c[i]) * grid.cell_volumes
        for j in range(i+1):
            k.cell_phi += a[i][j]*grid.deltaT* derivatives[j]
        Ax = lax.scatter_add(
            jnp.zeros(grid.n_cells),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * k.cell_phi[grid.edge_index[0]]
                    + FluxFf * k.cell_phi[grid.edge_index[1]]
                    + FluxVf,
                    -FluxFf * k.cell_phi[grid.edge_index[1]]
                    - FluxCf * k.cell_phi[grid.edge_index[0]]
                    - FluxVf,
                    FluxCb * k.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * k.bd_phi,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )

        RHS = (b_C-Ax)/(grid.cell_volumes * grid.rho)
        derivatives[i] = RHS


    # Calculate the final stage
    for i in range(nstage):
        y += grid.deltaT * b[i] * derivatives[i]
    new_timestep = foi.timestep + grid.deltaT
    foi_new = GridVariable(y, foi.bd_phi, new_timestep, foi.bc, foi.name).UpdateBoundaryPhi(
        grid
    )
    return foi_new,y

def crank_nicolson(
    FluxCf,
    FluxFf,
    FluxVf,
    FluxCb,
    FluxFb,
    FluxC,
    FluxC_old,
    FluxVf_old,
    foi: GridVariable,
    grid: Grid,
    source_function: Callable,
):

    a_C = lax.scatter_add(
        jnp.zeros(grid.n_cells),
        grid.face_owner_ud[..., None],
        jnp.concatenate((FluxCf, -FluxFf, FluxCb)),  # <N_f, 1> * <N_f, 3>
        grid.scalar_scatter_dim_num,
    )
    a_C = FluxC + a_C

    a_nb = (
        jnp.hstack((grid.edge_index, jnp.flip(grid.edge_index, axis=0))),
        jnp.concatenate((FluxFf, -FluxCf)),
    )

    source_Old = source_function(grid.cell_centers, foi.timestep)
    source = source_function(grid.cell_centers, foi.timestep + grid.deltaT)

    b_C = source_Old * grid.cell_volumes + source * grid.cell_volumes
    -foi.cell_phi * FluxC_old
    -FluxC * foi.cell_phi

    if grid.snGradSchemes == "orthogonal" or grid.snGradSchemes == "uncorrected":
        b_C -= 2 * lax.scatter_add(
            jnp.zeros(grid.n_cells),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]],
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]],
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )  # this is the equation residual
    
    if grid.snGradSchemes == "corrected":
        # raise NotImplementedError('Crank Nicolson + gauss linear corrected has issue now, fixing code bug')
        b_C -= lax.scatter_add(
            jnp.zeros(grid.n_cells),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]]
                    + FluxVf,
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]]
                    - FluxVf,
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        ) 
        
        b_C -= lax.scatter_add(
            jnp.zeros(grid.n_cells),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]]
                    +FluxVf_old,
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]]
                    -FluxVf_old,
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )  # this is the equation residual 
    
    return a_C, a_nb, b_C


def backward(
    FluxCf,
    FluxFf,
    FluxVf,
    FluxCb,
    FluxFb,
    FluxC,
    FluxC_old,
    FluxC_old_old,
    foi: GridVariable,
    foi_old:GridVariable,
    grid: Grid,
    source_function: Callable,
):

    a_C = lax.scatter_add(
        jnp.zeros(grid.n_cells),
        grid.face_owner_ud[..., None],
        jnp.concatenate((FluxCf, -FluxFf, FluxCb)),  # <N_f, 1> * <N_f, 3>
        grid.scalar_scatter_dim_num,
    )
    a_C = FluxC + a_C

    a_nb = (
        jnp.hstack((grid.edge_index, jnp.flip(grid.edge_index, axis=0))),
        jnp.concatenate((FluxFf, -FluxCf)),
    )

    source = source_function(grid.cell_centers, foi.timestep + grid.deltaT)

    b_C = source * grid.cell_volumes - FluxC_old * foi.cell_phi - FluxC_old_old * foi_old.cell_phi - FluxC * foi.cell_phi

    if grid.snGradSchemes == "orthogonal" or grid.snGradSchemes == "uncorrected":
        b_C -= lax.scatter_add(
            jnp.zeros((grid.n_cells)),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]],
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]],
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )  # this is the equation residual

    elif grid.snGradSchemes == "corrected":
        b_C -= lax.scatter_add(
            jnp.zeros((grid.n_cells)),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (
                    FluxCf * foi.cell_phi[grid.edge_index[0]]
                    + FluxFf * foi.cell_phi[grid.edge_index[1]]
                    + FluxVf,
                    -FluxFf * foi.cell_phi[grid.edge_index[1]]
                    - FluxCf * foi.cell_phi[grid.edge_index[0]]
                    - FluxVf,
                    FluxCb * foi.cell_phi[grid.owners[grid.n_I_faces :]]
                    + FluxFb * foi.bd_phi,
                    # + FluxVb,
                )
            ),  # <N_f, 1> * <N_f, 3>
            grid.scalar_scatter_dim_num,
        )  # this is the equation residual
    return a_C, a_nb, b_C
