import jax.numpy as jnp
import grids
from jax import lax

Array = grids.Array
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Grid = grids.Grid


def ComputeGradients(
    foi: GridVariable,
    grid: Grid,
) -> Array:
    """
    Green-Gauss with correction
    Assuming the bd fois are already imposed
    """

    c_foi, bd_foi = foi.cell_phi, foi.bd_phi
    p = grid.c_edge_index[0]
    q = grid.c_edge_index[1]
    gc = grid.ratio[:, 1]  # <N_c> Remember gc is dfF/dFC, not dfC/dFC
    # return jnp.mean(c_foi)
    # Linear initiation:
    internal_f_foi = gc * c_foi[p] + (1 - gc) * c_foi[q]  # <N_if, N_foi=1>
    # print('inter_f_foi iter0: ', type(internal_f_foi), internal_f_foi.shape)
    # print('bd_foi iter0: ', type(bd_foi), bd_foi.shape)
    # print('grid.c_vol:', grid.c_vol[:10])
    # print('face weights:',gc[:10])
    ## update the grad_C
    internal_c_grad_foi = (
        lax.scatter_add(
            jnp.zeros((grid.N_c, grid.dim)),
            grid.face_owner_ud[..., None],
            jnp.concatenate((internal_f_foi, internal_f_foi, bd_foi))[..., None]
            * grid.Sf_ud,  # <N_f, 1> * <N_f, 3>
            grid.vector_scatter_dim_num,
        )
        / grid.c_vol[..., None]
    )
    method = grid.fvSchemes["gradSchemes"]
    # <N_c, 3>
    if method == "Gauss linear":
        return internal_c_grad_foi
    elif method == "Gauss linear corrected":
        max_iter = grid.controlDict["gradSchemesMaxIter"]
        mae_internal_c_grad_foi = jnp.ones(grid.dim)
        N_iter = 0
        iter_pack = (internal_c_grad_foi, mae_internal_c_grad_foi, N_iter)

        def green_gauss_loop(iter_pack):
            internal_c_grad_foi, mae_internal_c_grad_foi, N_iter = iter_pack
            internal_f_foi_new = (
                internal_f_foi
                + gc
                * jnp.sum(
                    internal_c_grad_foi[p]
                    * (grid.f_center[: grid.N_if] - grid.C_pos[: grid.N_if]),
                    axis=-1,
                )
                + (1 - gc)
                * jnp.sum(
                    internal_c_grad_foi[q]
                    * (grid.f_center[: grid.N_if] - grid.F_pos[: grid.N_if]),
                    axis=-1,
                )
            )
            internal_c_grad_foi_new = (
                lax.scatter_add(
                    jnp.zeros((grid.N_c, grid.dim)),
                    grid.face_owner_ud[..., None],
                    jnp.concatenate((internal_f_foi_new, internal_f_foi_new, bd_foi))[
                        ..., None
                    ]
                    * grid.Sf_ud,  # <N_f, 1> * <N_f, 3>
                    grid.vector_scatter_dim_num,
                )
                / grid.c_vol[..., None]
            )
            # <N_c, 3>
            mae_internal_c_grad_foi = jnp.mean(
                abs(internal_c_grad_foi_new - internal_c_grad_foi), axis=0
            )
            # jax.debug.print("iter:{it}",it = N_iter)
            # jax.debug.print("error:{pv}",pv = jnp.max(abs(iter_pack[-2])))
            N_iter = N_iter + 1
            return (internal_c_grad_foi_new, mae_internal_c_grad_foi, N_iter)

        def criterion(iter_pack):
            # jax.debug.print("mae:{cp}",cp = jnp.max(iter_pack[-2]))
            # jax.debug.print("criterion1:{cp}",cp = (jnp.max(abs(iter_pack[-2])) > 1e-3))
            # jax.debug.print("criterion2:{cp}",cp = (iter_pack[-1]<=max_iter))
            return (jnp.max(iter_pack[-2]) > 1e-100) * (iter_pack[-1] <= max_iter)

        print("Compute gradients using the green gauss method")
        iter_pack = lax.while_loop(criterion, green_gauss_loop, iter_pack)
        print("gradient mae:", jnp.max(iter_pack[-2]))
        print(
            "Compute gradients using the green gauss method done with N_iter:",
            iter_pack[-1],
        )
        return iter_pack[0]
