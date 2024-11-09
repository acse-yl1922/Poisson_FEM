import jax.numpy as jnp
from base import grids
from jax import lax
Array = grids.Array 
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Grid = grids.Grid

from base.utils import ToUndirected
from base.interpolations import scalar_linear_interpolation, vector_linear_interpolation



def ComputeGradients(foi: GridVariable, grid: Grid) -> Array:
    gradscheme = grid.gradSchemes
    if gradscheme == "Gauss linear":
        return gauss_linear(foi, grid)
    elif gradscheme == "Gauss skewCorrected linear":
        return gauss_skewness_corrected_linear(foi, grid)
    else:
        raise NotImplementedError(f"gradScheme {gradscheme} not implmented")


def gauss_linear(foi: GridVariable, grid: Grid) -> Array:
    Sf_ud = ToUndirected(grid.faceSf, grid.n_I_faces, method="negative")
    phi_C, phi_bd = foi.cell_phi, foi.bd_phi
    # Linear initiation:
    internal_face_phi = scalar_linear_interpolation(phi_C, grid)
    cell_grad_phi = (
        lax.scatter_add(
            jnp.zeros((grid.n_cells, grid.dim)),
            grid.face_owner_ud[..., None],
            jnp.concatenate((internal_face_phi, internal_face_phi, phi_bd))[..., None]
            * Sf_ud,  # <N_f, 1> * <N_f, 3>
            grid.vector_scatter_dim_num,
        )
        / grid.cell_volumes[..., None]
    )
    return cell_grad_phi


def gauss_skewness_corrected_linear(foi: GridVariable, grid: Grid) -> Array:

    Sf_ud = ToUndirected(grid.faceSf, grid.n_I_faces, method="negative")
    phi_C, phi_bd = foi.cell_phi, foi.bd_phi

    def skewCorrectedVector(grid):
        ratio = jnp.sum(
            grid.faceSf[: grid.n_I_faces] * grid.faceCf[: grid.n_I_faces], axis=-1
        ) / jnp.sum(
            grid.faceSf[: grid.n_I_faces] * grid.faceCF[: grid.n_I_faces], axis=-1
        )
        return (
            grid.faceCf[: grid.n_I_faces]
            - ratio[..., None] * grid.faceCF[: grid.n_I_faces]
        )

    skewCorrectionVectors = skewCorrectedVector(grid)  # This part has been proved
    cell_grad_phi = gauss_linear(foi, grid)

    # Linear initiation:
    internal_face_phi = scalar_linear_interpolation(phi_C, grid)
    internal_face_grad_phi = vector_linear_interpolation(cell_grad_phi, grid)

    # correction_part
    internal_face_phi_corrected = internal_face_phi + jnp.sum(
        internal_face_grad_phi * skewCorrectionVectors, axis=-1
    )
    cell_grad_phi_corrected = (
        lax.scatter_add(
            jnp.zeros((grid.n_cells, grid.dim)),
            grid.face_owner_ud[..., None],
            jnp.concatenate(
                (internal_face_phi_corrected, internal_face_phi_corrected, phi_bd)
            )[..., None]
            * Sf_ud,  # <N_f, 1> * <N_f, 3>
            grid.vector_scatter_dim_num,
        )
        / grid.cell_volumes[..., None]
    )

    return cell_grad_phi_corrected


def gauss_loop_corrected_linear(foi: GridVariable, grid: Grid, max_iter) -> Array:
    #This implementation follows the method in book
    p = grid.edge_index[0]
    q = grid.edge_index[1]
    gc = 1 - grid.ratio[: grid.n_I_faces]
    Sf_ud = ToUndirected(grid.faceSf, grid.n_I_faces, method="negative")
    phi_C, phi_bd = foi.cell_phi, foi.bd_phi

    cell_grad_phi = gauss_linear(foi, grid)
    internal_face_phi = scalar_linear_interpolation(phi_C, grid)

    mae_internal_c_grad_foi = jnp.ones(grid.dim)
    N_iter = 0
    iter_pack = (cell_grad_phi, mae_internal_c_grad_foi, N_iter)

    def green_gauss_loop(iter_pack):
        cell_grad_phi, mae_internal_c_grad_foi, N_iter = iter_pack
        internal_phi_face_new = (
            internal_face_phi
            + gc * jnp.sum(cell_grad_phi[p] * (grid.faceCf[: grid.n_I_faces]), axis=-1)
            + (1 - gc)
            * jnp.sum(cell_grad_phi[q] * (grid.faceFf[: grid.n_I_faces]), axis=-1)
        )
        cell_grad_phi_new = (
            lax.scatter_add(
                jnp.zeros((grid.n_cells, grid.dim)),
                grid.face_owner_ud[..., None],
                jnp.concatenate((internal_phi_face_new, internal_phi_face_new, phi_bd))[
                    ..., None
                ]
                * Sf_ud,  # <N_f, 1> * <N_f, 3>
                grid.vector_scatter_dim_num,
            )
            / grid.cell_volumes[..., None]
        )
        # <N_c, 3>
        mae_internal_c_grad_foi = jnp.mean(
            abs(cell_grad_phi_new - cell_grad_phi), axis=0
        )
        N_iter = N_iter + 1
        return (cell_grad_phi_new, mae_internal_c_grad_foi, N_iter)

    def criterion(iter_pack):
        return (jnp.max(iter_pack[-2]) > 1e-100) * (iter_pack[-1] <= max_iter)

    print("Compute gradients using the green gauss method")
    iter_pack = lax.while_loop(criterion, green_gauss_loop, iter_pack)
    print("gradient mae:", jnp.max(iter_pack[-2]))
    print(
        "Compute gradients using the green gauss method done with N_iter:",
        iter_pack[-1],
    )
    return iter_pack[0]
