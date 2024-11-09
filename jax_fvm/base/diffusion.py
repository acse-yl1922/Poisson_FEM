from base import grids
import jax.numpy as jnp
from base.operators import ComputeGradients
from base.interpolations import vector_linear_interpolation

Array = grids.Array
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Grid = grids.Grid

def DiffusionFlux_linear(
    foi: GridVariable,
    grid: Grid,
):
    """
    Assemble the Flux Coefficients, linear part only
    """
    # calculate the internal edge contributions
    FluxCf = grid.gamma[: grid.n_I_faces] * grid.gDiff[: grid.n_I_faces]
    FluxFf = -grid.gamma[: grid.n_I_faces] * grid.gDiff[: grid.n_I_faces]

    # nonlinear part
    cell_grad_phi = ComputeGradients(foi, grid)
    internal_f_grad_foi = vector_linear_interpolation(cell_grad_phi, grid)
    FluxVf = -grid.gamma[: grid.n_I_faces] * jnp.sum(
        internal_f_grad_foi * grid.faceTf[: grid.n_I_faces], axis=-1
    )

    FluxCb_list = []
    FluxFb_list = []
    # FluxVb_list = []

    bd_values = grid.GetAllBoundaryValues(foi)
    
    # approximate gradient on the bd faces
    for i in range(len(grid.patch_names)):
        sid, eid = grid.patch_sids[i], grid.patch_eids[i]
        if foi.bc.bd_infos[i+1][1] == 'fixedValue':
            FluxCb_list.append(grid.gamma[sid:eid] * grid.gDiff[sid:eid])
            FluxFb_list.append(-grid.gamma[sid:eid] * grid.gDiff[sid:eid])

        elif foi.bc.bd_infos[i+1][1] == 'zeroGradient':
            FluxCb_list.append(jnp.zeros_like(bd_values[i]))
            FluxFb_list.append(jnp.zeros_like(bd_values[i]))
        else:
            # TBI
            pass
    FluxCb = jnp.concatenate(FluxCb_list)
    FluxFb = jnp.concatenate(FluxFb_list)
    # FluxVb = jnp.concatenate(FluxVb_list)
    return FluxCf, FluxFf, FluxVf, FluxCb, FluxFb
