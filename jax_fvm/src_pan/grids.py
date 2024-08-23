from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence, Tuple, Union

import jax
from jax import vmap, lax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

# TODO(jamieas): consider moving common types to a separate module.
# TODO(shoyer): consider adding jnp.ndarray?
Array = Union[np.ndarray, jax.Array]
IntOrSequence = Union[int, Sequence[int]]

# There is currently no good way to indicate a jax "pytree" with arrays at its
# leaves. See https://jax.readthedocs.io/en/latest/jax.tree_util.html for more
# information about PyTrees and https://github.com/google/jax/issues/3340 for
# discussion of this issue.
PyTree = Any


@dataclasses.dataclass(init=False, frozen=True)
class BoundaryConditions:
    """Base class for boundary conditions on a PDE variable.

    Attributes:
    -----------

    types: `types[i]` is a tuple specifying BC types for patch [i]
    0,1,2 indicates Dirichlet, Neuman, and Robin
    """

    bd_names: Tuple[str, ...]  # type for each patch
    bd_types: Tuple[str, ...]  # type for each patch
    bd_infos: Tuple[
        Tuple[str, int, str, Union[Callable, Array, float], int, int], ...
    ]  # all details

    def __init__(
        self,
        bd_infos: Tuple[Tuple[str, int, str, Union[Callable, Array, float]], ...],
        grid: Grid,
    ):
        """
        The input should be a list of ints indicating variable boundary conditions
        Inputs:
            Bd_info:
                str: name of the boundary
                int: type of the boundary Dirichlet 0, Neuman 1, Robin 2
                str: Uniform, Parabolic, Function, Readin
                Union[...]: Function, Array, Tuple of Arrays
                float: start face id
                float: end face id
        """
        # sort the boundareis as a sequence same to the grid
        temp_names = tuple(ele[0] for ele in bd_infos)
        indices = []
        for name in temp_names:
            indices.append(grid.patch_names.index(name))
        # print(indices)
        sorted_bd_infos = tuple(bd_infos[i] for i in indices)
        object.__setattr__(
            self, "bd_names", tuple(ele[0] for ele in sorted_bd_infos)
        )  # boundary types
        bd_dict = {0: "Dirichlet", 1: "Neuman", 2: "Robin"}
        object.__setattr__(
            self, "bd_types", tuple(bd_dict[ele[1]] for ele in sorted_bd_infos)
        )  # boundary types
        bd_infos_new = tuple(
            tuple([*sorted_bd_infos[i], grid.patch_sids[i], grid.patch_eids[i]])
            for i in range(len(temp_names))
        )
        object.__setattr__(self, "bd_infos", bd_infos_new)  # detail infos

    @property
    def GetNumOfBoundaries(self):
        return len(self.bd_names)


@register_pytree_node_class
@dataclasses.dataclass
class GridVariable:
    """Associates a GridArray with BoundaryConditions."""

    cell_phi: Array
    bd_phi: Array
    bc: BoundaryConditions
    name: str = "temp"
    # residual: float = 0.

    def tree_flatten(self):
        """Returns flattening recipe for GridVariable JAX pytree."""
        children = (self.cell_phi, self.bd_phi)
        aux_data = (self.bc,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Returns unflattening recipe for GridVariable JAX pytree."""
        return cls(*children, *aux_data)

    @property
    def dtype(self):
        return self.cell_phi.dtype

    @property
    def shape(self) -> Tuple[Tuple[int, ...], ...]:
        return self.cell_phi.shape, self.bd_phi.shape

    def GetBoundaryPhi(self, grid) -> Array:

        # bd_values = grid.bd_values[self.name]
        bd_values = grid.GetAllBoundaryValues(self.bc.bd_infos)
        bd_array = []
        # load in bdfoi
        for i in range(self.bc.GetNumOfBoundaries):
            _, bd_type, _, _, sid, eid = self.bc.bd_infos[i]
            if bd_type == 0:
                bd_array.append(bd_values[i])
            elif bd_type == 1:
                # approximate the gradient along CF
                bd_array.append(
                    self.cell_phi[grid.face_owner[sid:eid]]
                    + grid.dCF[sid:eid] * bd_values[i]
                )
            elif bd_type == 2:
                # TBI
                pass
        return jnp.array(jnp.concatenate(bd_array))  # phi values on the boundary

    def UpdateBoundaryPhi(self, grid) -> GridVariable:
        return GridVariable(
            self.cell_phi, self.GetBoundaryPhi(grid), self.bc, self.name
        )


GridVariableVector = Tuple[GridVariable, ...]


from tools import (
    ComputeFaceCenter,
    ComputeEfnormTf,
    ComputeFaceVector,
    ToUndirected,
    ComputeRatio,
)


# define the grid
# @dataclasses.dataclass(init=False)
@dataclasses.dataclass(init=False, frozen=True)
class Grid:
    """
    Class describing an unstructure grid

    Attributes
    ----------

    N_v (int): Number of vertices.
    N_c (int): Number of cells.
    dim (int): Dimension of the mesh.
    N_f (int): Total number of faces.
    N_if (int): Number of internal faces.
    N_bdf (int): Number of boundary faces.
    N_cbdf (int): Number of cell centers plus boundary face centers.
    patch_names (Tuple[str, ...]): Names of the patches.
    patch_types (Tuple[int, ...]): Types of the patches.
    patch_sids (Tuple[int, ...]): Starting indices of the patches.
    patch_eids (Tuple[int, ...]): Ending indices of the patches.
    """
    N_v: int
    N_c: int
    dim: int
    N_f: int
    N_if: int
    N_bdf: int
    N_cbdf: int
    patch_names: Tuple[str, ...]
    patch_types: Tuple[int, ...]
    patch_sids: Tuple[int, ...]
    patch_eids: Tuple[int, ...]

    def __init__(self, mesh_file=np.lib.npyio.NpzFile):
        """
        The input should be a npz file of all the information of a mesh
        """
        # read in mesh information
        for _, ele in enumerate(mesh_file):
            if "patch" in ele:
                object.__setattr__(self, ele, tuple(mesh_file[ele]))
            else:
                object.__setattr__(self, ele, jnp.asarray(mesh_file[ele]))    
        object.__setattr__(self, "N_v", self.pos.shape[0])  # num of vertices
        object.__setattr__(self, "dim", self.pos.shape[1])  # dimension of the mesh
        object.__setattr__(self, "N_c", self.c_pos.shape[0])  # num of cells
        object.__setattr__(self, "N_f", self.faces.shape[0])  # number of total faces
        object.__setattr__(
            self, "N_if", self.c_edge_index.shape[1]
        )  # number of internal faces
        object.__setattr__(
            self, "N_bdf", self.N_f - self.N_if
        )  # number of boundary faces
        object.__setattr__(
            self, "N_cbdf", self.N_c + self.N_bdf
        )  # number of cell centers + boundary face centers
        object.__setattr__(
            self, "patch_eids", self.patch_sids[1:] + (self.N_f,)
        )  # number of cell centers + boundary face centers

    @property
    def GetNumOfDim(self) -> int:
        """Returns the number of dimensions of this grid."""
        return self.dim

    @property
    def GetNumOfNodes(self) -> int:
        """Returns the number of nodes of this grid."""
        return self.N_v

    @property
    def GetNumOfCells(self) -> int:
        """Returns the number of cells of this grid."""
        return self.N_c

    def EvaluateOnMesh(self, fn: Callable[..., Array], region: str) -> Array:
        """
        Evaluates the function on the grid mesh.
        """
        if region == "cell":
            return fn(self.c_pos)
        if region == "cell_transient":
            frames = vmap(fn)(self.time_axis)
            return frames
        elif region == "boundary":
            return fn(self.f_center[self.N_if :])
        elif region == "cell and boundary":
            return fn(jnp.vstack(self.C_pos, self.f_center[self.N_if :]))
        elif region == "face":
            return fn(self.f_center)

    def SetUpSimulation(self, equation_name, **kwargs):
        """
        Inputs:
            equation_name: name of the equation
            **kwargs: params needed for the corresponding equation
        """
        self.Precompute()
        object.__setattr__(self, "equation", equation_name)
        if equation_name == "SteadyDiffusion":
            for key, value in kwargs.items():
                if key in ["source", "gt"]:
                    if type(value) == float or (
                        type(value) == Array and value.shape == (1,)
                    ):
                        myfield = value * jnp.ones(self.N_c)
                    elif type(value) == Array:
                        myfield = value
                    elif callable(value):
                        myfield = self.EvaluateOnMesh(value, "cell")
                    else:
                        raise Exception("input type is not supported")
                    object.__setattr__(self, key, myfield)

                elif "gamma" == key:  # we need gammas on the faces
                    if type(value) == float or (
                        type(value) == Array and value.shape == (1,)
                    ):
                        myfield = value * jnp.ones(self.N_f)
                    elif type(value) == Array:
                        myfield = value
                    elif callable(value):
                        myfield = self.EvaluateOnMesh(value, "face")
                    else:
                        raise Exception("input type is not supported")
                    object.__setattr__(self, key, myfield)
                else:
                    object.__setattr__(self, key, value)
        
        elif equation_name == "transient Diffusion":
            for key, value in kwargs.items():
                if "controlDict" == key:  # we need dt and 
                    startTime = value['startTime']
                    endTime = value['endTime']
                    deltaT = value['deltaT']
                    time_steps = jnp.arange(startTime, endTime, deltaT)
                    object.__setattr__(self, 'startTime', startTime)
                    object.__setattr__(self, 'endTime', endTime)
                    object.__setattr__(self, 'deltaT', deltaT)
                    # Define a function that adds the time column to c_pos
                    def add_time_axis(t):
                        return jnp.hstack((self.c_pos, t * jnp.ones((self.c_pos.shape[0], 1))))
                    
                    # Use vmap to vectorize the operation over all time steps
                    time_axis = vmap(add_time_axis)(time_steps)
                    # Store the result as an attribute
                    object.__setattr__(self, 'time_axis', time_axis)
                    
                elif key == "gt":
                    if type(value) == float or (
                        type(value) == Array and value.shape == (1,)
                    ):
                        myfield = jnp.ones(self.N_c)
                    elif type(value) == Array:
                        myfield = value
                    elif callable(value):
                        myfield = self.EvaluateOnMesh(value, "cell_transient")
                    else:
                        raise Exception("input type is not supported")
                    object.__setattr__(self, key, myfield)
                
                elif key == "source":
                    if type(value) == float or (
                        type(value) == Array and value.shape == (1,)
                    ):
                        myfield = value * jnp.ones(self.N_c)
                    elif type(value) == Array:
                        myfield = value
                    elif callable(value):
                        myfield = self.EvaluateOnMesh(value, "cell")
                    else:
                        raise Exception("input type is not supported")
                    object.__setattr__(self, key, myfield)
                
                elif "gamma" == key:  # we need gammas on the faces
                    if type(value) == float or (
                        type(value) == Array and value.shape == (1,)
                    ):
                        myfield = value * jnp.ones(self.N_f)
                    elif type(value) == Array:
                        myfield = value
                    elif callable(value):
                        myfield = self.EvaluateOnMesh(value, "face")
                    else:
                        raise Exception("input type is not supported")
                    object.__setattr__(self, key, myfield)

                elif "rho" == key:  # we need rho on the cell centre
                    if type(value) == float or (
                        type(value) == Array and value.shape == (1,)
                    ):
                        myfield = value * jnp.ones(self.N_c)
                    elif type(value) == Array:
                        myfield = value
                    elif callable(value):
                        myfield = self.EvaluateOnMesh(value, "cell")
                    else:
                        raise Exception("input type is not supported")
                    object.__setattr__(self, key, myfield)
                
                else:
                    object.__setattr__(self, key, value)

        elif equation_name == "Convection Diffusion":
            # TBI
            pass
        elif equation_name == "Navier Stokes":
            # TBI
            pass

    def GetAllBoundaryValues(self, bd_infos):
        bd_value_list = []
        for _, bd_info in enumerate(bd_infos):
            bd_value_list.append(self.GetBoundaryValues(bd_info))
        return tuple(bd_value_list)

    def GetBoundaryValues(self, bd_info):
        # assert bd_name in self.patch_names, "The boundary name is not detected"
        _, _, bd_info_type, content, sid, eid = bd_info
        if bd_info_type == "Uniform":
            return jnp.ones(eid - sid) * content
        elif bd_info_type == "Parabolic":
            # TBI
            pass
        elif bd_info_type == "Function":
            vmap_temp_func = vmap(content, inaxis=0)
            return vmap_temp_func(self.f_center[sid:eid])
        elif bd_info_type == "Readin":
            return jnp.load(content)

    def Precompute(self):
        p = self.c_edge_index[0]
        q = self.c_edge_index[1]
        # self.C_pos, self.F_pos = self.c_pos[p],self.c_pos[q]
        object.__setattr__(self, "f_center", ComputeFaceCenter(self.pos, self.faces))
        object.__setattr__(
            self, "Sf", ComputeFaceVector(self.pos, self.faces)
        )  # surface vector
        object.__setattr__(
            self, "Sf_ud", ToUndirected(self.Sf, self.N_if, method="negative")
        )  # including double internal face and bd face
        object.__setattr__(self, "area", jnp.linalg.norm(self.Sf, axis=1))
        object.__setattr__(
            self,
            "ratio",
            ComputeRatio(
                self.pos, self.c_pos, self.faces[: self.N_if], self.c_edge_index
            ),
        )
        object.__setattr__(
            self,
            "face_owner_ud",
            jnp.concatenate(
                (
                    p,
                    q,
                    self.face_owner[self.N_if :],
                ),
                axis=0,
            ),
        )  # including double internal face and bd face

        # we calculate all the Ef and Tf here,
        # self.Ef_norm, self.Tf = ComputeEfnormTf(self.c_pos,self.c_edge_index,self.Sf[:self.N_if])
        object.__setattr__(self, "C_pos", self.c_pos[self.face_owner])  # sender node
        object.__setattr__(
            self, "F_pos", jnp.vstack((self.c_pos[q], self.f_center[self.N_if :]))
        )  # receiver node
        Ef_norm, Tf = ComputeEfnormTf(self.C_pos, self.F_pos, self.Sf)
        object.__setattr__(self, "Ef_norm", Ef_norm)
        object.__setattr__(self, "Tf", Tf)  #
        object.__setattr__(
            self, "dCF", jnp.linalg.norm(self.C_pos - self.F_pos, axis=1)
        )  #
        CF = self.F_pos - self.C_pos
        Cb_dist = jnp.sum(
            CF[self.N_if :] * self.Sf[self.N_if :] / self.area[self.N_if :][..., None],
            axis=1,
        )
        object.__setattr__(
            self, "dCF_dist", jnp.concatenate((self.dCF[: self.N_if], Cb_dist))
        )  #

        object.__setattr__(
            self, "eCF", (self.C_pos - self.F_pos) / self.dCF[..., None]
        )  #
        object.__setattr__(
            self, "gDiff_f", self.Ef_norm / self.dCF_dist
        )  # this is relevant to the face index
        # scatter parameters
        scalar_scatter_dim_num = lax.ScatterDimensionNumbers((), (0,), (0,))
        vector_scatter_dim_num = lax.ScatterDimensionNumbers(
            (1,), (0,), (0,)  # window of update  # insert of update
        )  # map scatter to operand
        object.__setattr__(self, "scalar_scatter_dim_num", scalar_scatter_dim_num)  #
        object.__setattr__(self, "vector_scatter_dim_num", vector_scatter_dim_num)  #
