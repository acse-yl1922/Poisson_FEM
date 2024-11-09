import re
import pyvista as pv
import jax.numpy as jnp
import jax

# jax.config.update("jax_enable_x64", True)


# define the function that calculates coefficients given triangle and two cell centers
def DotProduct(a, b):
    return jnp.sum(a * b, axis=-1)


def ComputeEfnormTf(region, grid):
    """ """
    method = region.foamDictionary['fvSchemes']['snGradSchemes']
    E_normed = grid.faceCF / jnp.linalg.norm((grid.faceCF), axis=-1, keepdims=True)

    # This part of the code is following the style of openfoam
    if method == "orthogonal":

        Ef_norm = jnp.linalg.norm(grid.faceSf, axis=-1)
        Tf = jnp.zeros((grid.n_I_faces + grid.n_B_faces, 3))

    elif method == "uncorrected":
        # In openfoam 'corrected' is the   'over-relaxed' approach in the book
        Ef_norm = (
            jnp.sum(grid.faceSf * grid.faceSf, axis=-1)
            / jnp.sum(grid.faceSf * E_normed, axis=-1)
        )[: grid.n_I_faces]

        Ef_norm_bd = grid.face_areas[grid.n_I_faces :]

        Ef_norm = jnp.concatenate([Ef_norm, Ef_norm_bd])
        
        Tf = jnp.zeros((grid.n_I_faces + grid.n_B_faces, 3))
    
    elif method == "corrected" or method == "over_relaxed":
    # In openfoam 'corrected' is the   'over-relaxed' approach in the book
        Ef_norm = (
            jnp.sum(grid.faceSf * grid.faceSf, axis=-1)
            / jnp.sum(grid.faceSf * E_normed, axis=-1)
        )[: grid.n_I_faces]

        Ef_norm_bd = grid.face_areas[grid.n_I_faces :]

        Ef_norm = jnp.concatenate([Ef_norm, Ef_norm_bd])

        Ef = Ef_norm[..., None] * E_normed

        n = grid.faceSf / jnp.linalg.norm((grid.faceSf), axis=-1, keepdims=True)

        cosTheta = jnp.sum(grid.faceSf * Ef, axis=-1) / (Ef_norm * grid.face_areas)

        Tf = (n - E_normed / cosTheta[..., None]) * grid.face_areas[..., None]
    
    # Here we keep the interface for other 2 correction approach metioned in book
    elif method == "minimum_correction":
        Ef_norm = jnp.sum(grid.faceSf*E_normed, axis = -1)[: grid.n_I_faces]
        Ef_norm_bd = grid.face_areas[grid.n_I_faces :]
        Ef_norm = jnp.concatenate([Ef_norm, Ef_norm_bd])
        Ef = Ef_norm[..., None] * E_normed
        Tf = grid.faceSf-Ef
    
    elif method == "normal_correction":
        Ef_norm = jnp.linalg.norm(grid.faceSf, axis= -1)[: grid.n_I_faces]
        Ef_norm_bd = grid.face_areas[grid.n_I_faces :]
        Ef_norm = jnp.concatenate([Ef_norm, Ef_norm_bd])
        Ef = Ef_norm[...,None]*E_normed
        Tf = grid.faceSf-Ef
    
    return Ef_norm, Tf


def ToUndirected(foi, N_inter, method="flip"):
    """
    Converts a directed graph's features or edges into an undirected form
    using one of several methods.

    Parameters
    ----------
    foi : ndarray
        An array of shape (M, ...) representing the features or edges of a graph.
        The first `N_inter` entries are treated as directed edges that need to be
        transformed into undirected form. The remaining entries (`foi[N_inter:]`)
        are appended to the result without modification.

    N_inter : int
        The number of directed interactions or connections to be transformed.
        The first `N_inter` elements of `foi` are considered as directed edges.

    method : str, optional
        The method used to make the graph undirected. Options are:
        - "negative": Returns the original edges followed by their negative counterparts.
        - "duplicate": Returns the original edges followed by a duplicate of those edges.
        - "flip": Returns the original edges followed by their flipped counterparts.
        The default is "flip".

    Returns
    -------
    ndarray
        An array of shape (2 * N_inter + (M - N_inter), ...) containing the transformed
        edges or features. The output includes the original `foi[:N_inter]`, the modified
        version based on the chosen method, and the remaining `foi[N_inter:]`.

    """
    if method == "negative":
        return jnp.concatenate((foi[:N_inter], -foi[:N_inter], foi[N_inter:]), axis=0)
    elif method == "duplicate":
        return jnp.concatenate((foi[:N_inter], foi[:N_inter], foi[N_inter:]), axis=0)
    elif method == "flip":
        return jnp.concatenate(
            (foi[:N_inter], jnp.flip(foi[:N_inter], axis=-1), foi[N_inter:]), axis=0
        )


def diag_coordinates_tensor(grid):
    "Create diag position indices for 3d grad(phi) operator"
    # Create an array for each index i (0, 1, 2) and stack them together
    i_coords = jnp.repeat(jnp.arange(3), grid.n_cells)  # Repeats [0, 1, 2] each n times

    # Create arrays for diagonal indices (j, j)
    j_coords = jnp.tile(
        jnp.arange(grid.n_cells), 3
    )  # Repeats [0, 1, ..., n-1] three times

    # Stack coordinates along the second axis to form the coordinates array
    coords = jnp.stack([i_coords, j_coords, j_coords], axis=1)
    return coords


def off_diag_coordinates_tensor(grid):
    "Create off-diag position indices for 3d grad(phi) operator"
    # Define edge_index
    edge_index = jnp.hstack((grid.edge_index, jnp.flip(grid.edge_index, axis=0)))

    # Get the number of edges
    num_edges = edge_index.shape[1]

    # Generate the first index (0, 1, 2) repeated for each edge
    i_coords = jnp.repeat(jnp.arange(3), num_edges)

    # Repeat edge_index for each of the three matrices
    senders = jnp.tile(edge_index[0], 3)
    receivers = jnp.tile(edge_index[1], 3)

    # Stack the coordinates to form the coords array
    coords = jnp.stack([i_coords, senders, receivers], axis=1)
    return coords
