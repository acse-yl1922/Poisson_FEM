import grids
import jax.numpy as jnp
from operators import ComputeGradients
from interpolations import InterpolateGradientsFromCellToInteriorFace

Array = grids.Array
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Grid = grids.Grid


def DiffusionFlux_linear(
    foi: GridVariable,
    grid: Grid,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """
    Assembles diffusion flux coefficients for the given field over the grid.

    Parameters
    -----------
    foi : GridVariable
        The field variable with values and boundary conditions.

    grid : Grid
        The grid structure with geometrical data and boundary information.

    Returns
    --------
    FluxCf, FluxFf, FluxVf : Array
        Linear and nonlinear fluxes at internal faces.

    FluxCb, FluxFb, FluxVb : Array
        Linear and nonlinear fluxes at boundary faces.
    """

    # calculate the internal edge contributions
    # reference: 8.6.7 Algebraic Equation for Non-orthogonal Meshes
    FluxCf = grid.gamma[: grid.N_if] * grid.gDiff_f[: grid.N_if]
    FluxFf = -grid.gamma[: grid.N_if] * grid.gDiff_f[: grid.N_if]

    # nonlinear part
    # reference: 8.6.7 Algebraic Equation for Non-orthogonal Meshes
    internal_c_grad_foi = ComputeGradients(foi, grid)
    internal_f_grad_foi = InterpolateGradientsFromCellToInteriorFace(
        foi, internal_c_grad_foi, grid
    )
    FluxVf = -grid.gamma[: grid.N_if] * jnp.sum(
        internal_f_grad_foi * grid.Tf[: grid.N_if], axis=-1
    )

    # boundary conditions
    FluxCb_list = []
    FluxFb_list = []
    FluxVb_list = []

    bd_values = grid.GetAllBoundaryValues(foi.bc.bd_infos)
    # approximate gradient on the bd faces
    for i in range(len(grid.patch_names)):
        sid, eid = grid.patch_sids[i], grid.patch_eids[i]
        if grid.patch_types[i] == 0:
            FluxCb_list.append(grid.gamma[sid:eid] * grid.gDiff_f[sid:eid])
            FluxFb_list.append(-grid.gamma[sid:eid] * grid.gDiff_f[sid:eid])
            # approximate graphi_b like so:

            grad_b_foi = (
                internal_c_grad_foi[grid.face_owner[sid:eid]]
                - jnp.sum(
                    internal_c_grad_foi[grid.face_owner[sid:eid]] * grid.eCF[sid:eid],
                    axis=-1,
                )[..., None]
                * grid.eCF[sid:eid]
                + (
                    (bd_values[i] - foi.cell_phi[grid.face_owner[sid:eid]])
                    / grid.dCF[sid:eid]
                )[..., None]
                * grid.eCF[sid:eid]
            )
            # now we can calculate the cross diffusion terms at the boudnary
            FluxVb_list.append(
                -grid.gamma[sid:eid] * jnp.sum(grad_b_foi * grid.Tf[sid:eid], axis=-1)
            )
        elif grid.patch_types[i] == 1:
            FluxCb_list.append(jnp.zeros_like(bd_values[i]))
            FluxFb_list.append(jnp.zeros_like(bd_values[i]))
            FluxVb_list.append(bd_values[i] * grid.area[sid:eid])
        else:
            # TBI
            pass

    FluxCb = jnp.concatenate(FluxCb_list)
    FluxFb = jnp.concatenate(FluxFb_list)
    FluxVb = jnp.concatenate(FluxVb_list)

    return FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb


def TransientFlux(
    grid: Grid,
) -> tuple[Array, Array, Array]:
    """
    Assembles the transient flux coefficients based on the grid properties.

    Returns
    -------
    
    FluxC
        Coefficient for the current time step.
    
    FluxC_old
        Coefficient for the previous time step.
    
    FluxV
        Additional transient flux term (zero in this case).
    """

    # Local fluxes
    # reference: 13.3 First Order Transient Schemes
    FluxC = grid.c_vol * grid.rho / grid.deltaT
    FluxC_old = -grid.c_vol * grid.rho / grid.deltaT
    FluxV = jnp.zeros_like(FluxC)
    return FluxC, FluxC_old, FluxV
