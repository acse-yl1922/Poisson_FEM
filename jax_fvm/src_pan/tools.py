import jax.numpy as jnp


# define the function that calculates coefficients given triangle and two cell centers
def DotProduct(a, b):
    return jnp.sum(a * b, axis=-1)


def ComputeFaceVector(vpos, faces):
    """
    Computes the normalized normal vectors for a given set of faces in a mesh using cross product,
    supporting both 2D and 3D cases.

    Parameters
    ----------
    vpos : ndarray
        An array of shape (Nv, Ndim) representing the coordinates of nodes, where
        Nv is the number of nodes and Ndim is the dimensionality (e.g., 2D or 3D).

    faces : ndarray
        An array of shape (Nf, 2) for 2D or (Nf, 3) for 3D, representing the indices
        of nodes that form each face, where Nf is the number of faces. Each face is
        defined by either two nodes (2D) or three nodes (3D).

    Returns
    -------
    ndarray
        An array of shape (Nf, Ndim) containing the normalized normal vectors for each face.

    Examples
    --------
    3D Example:
    >>> vpos = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> faces = jnp.array([[0, 1, 2]])
    >>> ComputeFaceVector(vpos, faces)
    Array([[0.5, 0.5, 0.5]], dtype=float32)

    2D Example:
    >>> vpos = jnp.array([[1, 0], [0, 1], [0, 0]])
    >>> faces = jnp.array([[0, 1], [1, 2], [2, 0]])
    >>> ComputeFaceVector(vpos, faces)
    Array([[ 0.5,  0.5],
           [-0.5,  0. ],
           [ 0. , -0.5]], dtype=float32)
    """
    Ts = vpos[faces]
    if vpos.shape[1] == 3:  # 3D case
        a = Ts[:, 1, :] - Ts[:, 0, :]
        b = Ts[:, 2, :] - Ts[:, 0, :]
        Sf = jnp.hstack(
            (
                (a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1])[..., None],
                (a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2])[..., None],
                (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])[..., None],
            )
        )
        return Sf / 2
    elif vpos.shape[1] == 2:  # 2D case
        # In 2D, the normal vector is perpendicular to the line segment
        a = Ts[:, 1] - Ts[:, 0]
        Sf = jnp.hstack((a[:, 1][..., None], -a[:, 0][..., None]))  # Rotate 90 degrees
        return Sf / 2
    else:
        raise ValueError("vpos must have 2 or 3 dimensions.")


def ComputeFaceNormal(vpos, faces):
    """
    inputs:
        vpos:    <Nv, Ndim>
        faces:   <Nf, 3> can include bd faces
    """
    Sf = ComputeFaceVector(vpos, faces)
    return Sf / jnp.linalg.norm(Sf, axis=1, keepdims=True)


def ComputeFaceArea(vpos, faces):
    """
    inputs:
        vpos:    <Nv, Ndim>
        faces:   <Nf, 3> can include bd faces
    """
    Sf = ComputeFaceVector(vpos, faces)
    return jnp.linalg.norm(Sf, axis=1)


def ComputeFaceCenter(vpos, faces):
    """
    Computes the position of face centers for a given set of faces.

    Parameters
    ----------
    vpos : ndarray
        An array of shape (Nv, Ndim) representing the coordinates of nodes, where
        Nv is the number of nodes and Ndim is the dimensionality (e.g., 2D or 3D).

    faces : ndarray
        An array of shape (Nf, 2) for 2D or (Nf, 3) for 3D, representing the indices
        of nodes that form each face, where Nf is the number of faces. Each face is
        defined by either two nodes (2D) or three nodes (3D).

    Returns
    -------
    ndarray
        An array of shape (Nf, Ndim) containing the coordinates of the center of each face.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> ComputeFaceCenter(jnp.array([[1, 0], [0, 1], [0, 0]]), jnp.array([[0, 1], [1, 2], [2, 0]]))
    Array([[0.5, 0.5],
           [0. , 0.5],
           [0.5, 0. ]], dtype=float32)
    """
    Ts = vpos[faces]
    return jnp.mean(Ts, axis=1)


def ComputeRatio(vpos, cpos, internal_faces, c_edge_index):
    """
    inputs:
        vpos:    <Nv, Ndim>
        cpos:    <Nc, Ndim>
        internal_faces:   <Nf, 3>  Do not include bd faces
        c_edge_index: <2, Ne>
    """
    Ps = cpos[c_edge_index.T]
    Ts = vpos[internal_faces]
    N_normed = ComputeFaceNormal(vpos, internal_faces)
    ratio = DotProduct(Ps[:, 0] - Ts[:, 0, :], N_normed) / DotProduct(
        Ps[:, 0] - Ps[:, 1], N_normed
    )
    ratios = jnp.hstack((ratio[..., None], 1 - ratio[..., None]))
    return ratios


def ComputeFaceIntersectPoint(cpos, c_edge_index, ratio):
    """
    inputs:
        cpos:    <Nc, Ndim>
        c_edge_index: <2, Ne>  Do not include bd faces
        ratios: <Ne,2>
    """
    Ps = cpos[c_edge_index.T]
    Face_intersect = (Ps[:, 1] - Ps[:, 0]) * ratio[..., None] + Ps[:, 0]
    return Face_intersect


def ComputeEfnormTf(cpos, f_pos, Sf, method="normal correction"):
    """
    inputs:
        cpos:    <N_bd_c, Ndim>   Do not include bd faces
        f_pos:    <N_bd_c, Ndim>   Do not include bd faces
        Sf:    <N_bd_c, Ndim>   Do not include bd faces
    """

    E_normed = (f_pos - cpos) / jnp.linalg.norm((f_pos - cpos), axis=-1, keepdims=True)
    if method == "minimum correction":
        Ef_norm = jnp.sum(Sf * E_normed, axis=-1)
        Ef = Ef_norm[..., None] * E_normed
        Tf = Sf - Ef
    elif method == "normal correction":
        Ef_norm = jnp.linalg.norm(Sf, axis=-1)
        Ef = Ef_norm[..., None] * E_normed
        Tf = Sf - Ef
    elif method == "over-relaxed correction":
        return NotImplementedError("over-relaxed correction method not implemented yet")

    return Ef_norm, Tf


def ComputeEfnormTf_Dirichlet(bd_cpos, bd_f_pos, bd_Sf):
    """
    inputs:
        bd_cpos:    <N_bd_c, Ndim>   Do not include bd faces
        bd_f_pos:    <N_bd_c, Ndim>   Do not include bd faces
        bd_Sf:    <N_bd_c, Ndim>   Do not include bd faces
    """
    # E_normed = (bd_f_pos - bd_cpos) / jnp.linalg.norm(
    #     (bd_f_pos - bd_cpos), axis=-1, keepdims=True
    # )
    # Ef_norm = jnp.sum(Sf * E_normed, axis=-1)
    # Ef = Ef_norm[..., None] * E_normed
    # Tf = bd_Sf - Ef
    # return Ef_norm, Tf
    return NotImplementedError("Method Not Implemented")


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
