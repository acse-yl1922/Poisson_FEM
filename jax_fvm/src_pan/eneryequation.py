import grids
import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
from operators import ComputeGradients
from interpolations import InterpolateGradientsFromCellToInteriorFace

Array = grids.Array
# GridArray = grids.GridArray
# GridArrayVector = grids.GridArrayVector
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


def CalculateConvectionFlux(
    foi: GridVariable,
    grid: Grid,
):
    """
    Assemble the Flux Coefficients
    Here we just take in a given velocity field named as grid.velocity_f, where "_f" means the values are on the faces
    """
    # use the velocity field to calcualte rho

    mdof_f = grid.rho * grid.Sf * grid.velocity_f

    # implementing upwind scheme
    # calculate the internal edge contributions
    FluxCf = jnp.max(jnp.hstack((mdof_f[: grid.N_if, None], jnp.zeros(grid.N_if))))
    FluxFf = -jnp.max(-jnp.hstack((mdof_f[: grid.N_if, None], jnp.zeros(grid.N_if))))

    FluxVf = jnp.zeros(grid.N_if)

    FluxCb_list = []
    FluxFb_list = []
    FluxVb = jnp.zeros(grid.N_bdf)

    bd_values = grid.GetAllBoundaryValues(foi.bc.bd_infos)
    # approximate gradient on the bd faces
    for i in range(len(grid.patch_names)):
        sid, eid = grid.patch_sids[i], grid.patch_eids[i]
        if grid.patch_types[i] == 0:
            FluxCb_list.append(
                jnp.max(jnp.hstack((mdof_f[sid:eid, None], jnp.zeros(eid - sid))))
            )
            FluxFb_list.append(
                -jnp.max(-jnp.hstack((mdof_f[sid:eid, None], jnp.zeros(eid - sid))))
            )
        elif grid.patch_types[i] == 1:
            FluxCb_list.append(
                jnp.max(jnp.hstack((mdof_f[sid:eid, None], jnp.zeros(eid - sid))))
            )
            FluxFb_list.append(
                -jnp.max(-jnp.hstack((mdof_f[sid:eid, None], jnp.zeros(eid - sid))))
            )
        else:
            # TBI
            pass

    FluxCb = jnp.concatenate(FluxCb_list)
    FluxFb = jnp.concatenate(FluxFb_list)

    # implementing SOU scheme
    # first of all we should calculate the dot product of the velocity and the surface vector
    iUpwind = grid.c_edge_index[(jnp.arange(grid.N_if), jnp.where(mdof_f >= 0, 0, 1))]
    # iDownwind = grid.c_edge_index[(jnp.arange(grid.N_if), jnp.where(mdof_f<0,0,1))]

    # nonlinear part
    internal_c_grad_foi = ComputeGradients(foi, grid)
    internal_f_grad_foi = InterpolateGradientsFromCellToInteriorFace(
        foi, internal_c_grad_foi, grid
    )
    # Calculate deferred correction
    dCFUpwind = grid.f_center[: grid.N_if] - grid.c_pos[iUpwind]
    FluxVf += mdof_f[: grid.N_if] * jnp.sum(
        2 * internal_c_grad_foi[iUpwind] * internal_f_grad_foi, dCFUpwind, axis=1
    )

    return FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb


def AssembleMatrixForm(
    FluxCf, FluxFf, FluxVf, FluxCb, FluxFb, FluxVb, foi: GridVariable, grid: Grid
):

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


from functools import partial


@partial(jax.jit, static_argnums=(1,))
def SolveOneIteration_jit(
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
    return foi_new, (foi_new, residual)


def SolveOneIteration(
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
    return foi_new, (foi_new, residual)


def PlotPoisson(
    foi: GridVariable, grid: Grid, output_dir=None, save_interval=20, max_time_iter=100
):
    residuals = []

    if output_dir:
        # initial guess
        plot_out_solution_compare(
            foi.cell_phi, grid, output_dir + "/meshes/solution_iter{:d}.jpg".format(0)
        )

    for i in range(max_time_iter):
        foi, _, res, _ = SolveOneIteration(foi, grid)
        residuals.append(jnp.max(abs(res)))
        print("equation residual: ", residuals[-1])
        print("peek solution: ", foi.cell_phi[:10])
        if output_dir:
            if (i) % save_interval == 0:
                # plot out the solution:
                # plot_out_solution(foi.cell_phi , grid, output_dir+'/meshes/solution_iter{:d}.png'.format(i))
                plot_out_solution_compare(
                    foi.cell_phi,
                    grid,
                    output_dir + "/meshes/solution_iter{:d}.jpg".format(i + 1),
                )
                # plot_out_solution_compare(foi.cell_phi, grid, output_dir+'/meshes/solution_iter{:d}.png'.format(i))
                # plot out the residual:
                plot_out_error(
                    {"Equations": residuals},
                    output_dir + "/figs/solution_iter{:d}.jpg".format(i + 1),
                )
                # plot_out_error({'Equations':residuals}, output_dir+'/figs/solution_iter{:d}.png'.format(i))


from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import pylab

params = {
    "legend.fontsize": 18,
    "axes.labelsize": 25,
    "axes.titlesize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
}
pylab.rcParams.update(params)


def plot_out_solution(solution, grid, output_filename=None):
    """
    plot the solution
    Inputs:
        solution: internal cell foi
    """
    # regular points:
    mx, my, mz = jnp.meshgrid(
        jnp.linspace(0, 1, 50), jnp.linspace(0, 1, 50), jnp.linspace(0, 1, 50)
    )
    mxyz = jnp.concatenate((mx[..., None], my[..., None], mz[..., None]), axis=-1)
    # interpolate
    fig, ax = plt.subplots(1, 3, figsize=(30, 8))
    # lw=1
    cx = 10
    cy = 20
    cz = 30
    inter_foi_yz = griddata(grid.c_pos, solution, mxyz[cx, :, :], method="linear")
    inter_foi_xz = griddata(grid.c_pos, solution, mxyz[:, cy, :], method="linear")
    inter_foi_xy = griddata(grid.c_pos, solution, mxyz[:, :, cz], method="linear")
    data = [inter_foi_yz, inter_foi_xz, inter_foi_xy]

    imx = ax[0].imshow(data[0], label="yz", extent=(0, 1, 0, 1))
    imy = ax[1].imshow(data[1], label="xz", extent=(0, 1, 0, 1))
    imz = ax[2].imshow(data[2], label="xy", extent=(0, 1, 0, 1))
    cbarx = fig.colorbar(imx)
    cbary = fig.colorbar(imy)
    cbarz = fig.colorbar(imz)
    ax[0].set_xlabel("z")
    ax[0].set_ylabel("x")
    ax[0].set_title("solution @ x = " + str(cx / 50))
    ax[1].set_xlabel("z")
    ax[1].set_ylabel("y")
    ax[1].set_title("solution @ y = " + str(cy / 50))
    ax[2].set_xlabel("y")
    ax[2].set_ylabel("x")
    ax[2].set_title("solution @ z = " + str(cz / 50))
    if output_filename:
        fig.savefig(output_filename, bbox_inches="tight")


def plot_out_solution_compare(solution, grid, output_filename=None):
    """
    plot the solution
    Inputs:
        solution: internal cell foi
    """
    # regular points:
    mx, my, mz = jnp.meshgrid(
        jnp.linspace(0, 1, 50), jnp.linspace(0, 1, 50), jnp.linspace(0, 1, 50)
    )
    mxyz = jnp.concatenate((mx[..., None], my[..., None], mz[..., None]), axis=-1)
    # interpolate
    fig, ax = plt.subplots(3, 3, figsize=(30, 24))
    # lw=1
    cx = 10
    cy = 20
    cz = 30
    solution_inter_foi_yz = griddata(
        grid.c_pos, solution, mxyz[cx, :, :], method="linear"
    )
    solution_inter_foi_xz = griddata(
        grid.c_pos, solution, mxyz[:, cy, :], method="linear"
    )
    solution_inter_foi_xy = griddata(
        grid.c_pos, solution, mxyz[:, :, cz], method="linear"
    )

    gt_inter_foi_yz = griddata(grid.c_pos, grid.gt_foi, mxyz[cx, :, :], method="linear")
    gt_inter_foi_xz = griddata(grid.c_pos, grid.gt_foi, mxyz[:, cy, :], method="linear")
    gt_inter_foi_xy = griddata(grid.c_pos, grid.gt_foi, mxyz[:, :, cz], method="linear")

    diff_inter_foi_yz = griddata(
        grid.c_pos, solution - grid.gt_foi, mxyz[cx, :, :], method="linear"
    )
    diff_inter_foi_xz = griddata(
        grid.c_pos, solution - grid.gt_foi, mxyz[:, cy, :], method="linear"
    )
    diff_inter_foi_xy = griddata(
        grid.c_pos, solution - grid.gt_foi, mxyz[:, :, cz], method="linear"
    )

    data = [
        [solution_inter_foi_yz, solution_inter_foi_xz, solution_inter_foi_xy],
        [gt_inter_foi_yz, gt_inter_foi_xz, gt_inter_foi_xy],
        [diff_inter_foi_yz, diff_inter_foi_xz, diff_inter_foi_xy],
    ]
    data_labels = ["solution", "gt", "difference"]
    for i in range(len(data)):
        imx = ax[i, 0].imshow(data[i][0], label="yz", extent=(0, 1, 0, 1))
        imy = ax[i, 1].imshow(data[i][1], label="xz", extent=(0, 1, 0, 1))
        imz = ax[i, 2].imshow(data[i][2], label="xy", extent=(0, 1, 0, 1))
        cbarx = fig.colorbar(imx)
        cbary = fig.colorbar(imy)
        cbarz = fig.colorbar(imz)
        ax[i, 0].set_xlabel("z")
        ax[i, 0].set_ylabel("x")
        ax[i, 0].set_title(data_labels[i] + "@ x = " + str(cx / 50))
        ax[i, 1].set_xlabel("z")
        ax[i, 1].set_ylabel("y")
        ax[i, 1].set_title(data_labels[i] + "@ y = " + str(cy / 50))
        ax[i, 2].set_xlabel("y")
        ax[i, 2].set_ylabel("x")
        ax[i, 2].set_title(data_labels[i] + "@ z = " + str(cz / 50))

    if output_filename:
        fig.savefig(output_filename, bbox_inches="tight")


def plot_out_error(errors_dict, output_filename=None):
    """
    plot the residual curve
    Inputs:
        errors_dict: should be dict containing the name and the values of the residuals
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    # lw=1
    for key, value in errors_dict.items():
        ax.plot(value, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Residual")
    leg = ax.legend(loc="upper right", frameon=True)
    leg.get_frame().set_edgecolor("black")
    if output_filename:
        fig.savefig(output_filename, bbox_inches="tight")
