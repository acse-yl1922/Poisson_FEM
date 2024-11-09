from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence, Tuple, Union
from base.region import Region

import jax
from jax import vmap, lax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

from base.utils import ComputeEfnormTf

# TODO(jamieas): consider moving common types to a separate module.
# TODO(shoyer): consider adding jnp.ndarray?
Array = Union[np.ndarray, jax.Array]
IntOrSequence = Union[int, Sequence[int]]

# There is currently no good way to indicate a jax "pytree" with arrays at its
# leaves. See https://jax.readthedocs.io/en/latest/jax.tree_util.html for more
# information about PyTrees and https://github.com/google/jax/issues/3340 for
# discussion of this issue.
PyTree = Any

jax.config.update("jax_enable_x64", True)

@dataclasses.dataclass(init=False, frozen=True)
class BoundaryConditions:
    """Base class for boundary conditions on a PDE variable.
    Attributes:
    types: `types[i]` is a tuple specifying BC types for patch [i]
    0,1,2 indicates Dirichlet, Neuman, and Robin
    """ 
    bd_infos: Tuple[Tuple[str, str,Callable[[Array], Array]], ...]
    def __init__(self,
                 bd_infos: Tuple[Tuple[str, str,Callable[[Array], Array]], ...]
                 ):
        '''
        The input should be a list of ints indicating variable boundary conditions
        Inputs: 
            Bd_info:
                str: name of the boundary
                str: bd type name in the openfoam sense 
                func: function that determines the bd values. 
        '''
        object.__setattr__(self, 'bd_infos',bd_infos) # detail infos 
    @property
    def GetNumOfBoundaries(self):
        return len(self.bd_infos)

@register_pytree_node_class
@dataclasses.dataclass
class GridVariable:
    """Associates a GridArray with BoundaryConditions.
    """
    cell_phi: Array
    bd_phi: Array
    timestep: float
    bc: BoundaryConditions
    name: str = 'temp'
    # residual: float = 0.
    
    def tree_flatten(self):
        """Returns flattening recipe for GridVariable JAX pytree."""
        children = (self.cell_phi, self.bd_phi, self.timestep)
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
        return self.cell_phi.shape, self.bd_phi.shape, len(self.timestep)

    def InitializeInterior(self, region: Region):
        temp_func = self.bc.bd_infos[0][-1]
        if 'vol' in region.fluid[self.name]['theType']: # volume variable 
            # print('debug:', temp_func(region.mesh.cell_centers).shape)
            # print('debug2:', GridVariable(temp_func(region.mesh.cell_centers),self.bd_phi,self.bc, self.name).shape)
            # print('debug3:', GridVariable(temp_func(region.mesh.cell_centers),self.bd_phi,self.bc, self.name))
            return GridVariable(temp_func(region.mesh.cell_centers),self.bd_phi, self.timestep, self.bc, self.name)
        elif 'surface' in region.fluid[self.name]['theType']: # surface variable 
            return GridVariable(temp_func(region.mesh.face_centers[:region.mesh.n_I_faces]),self.bd_phi, self.timestep, self.bc, self.name)

    def GetBoundaryPhi(self,grid: Grid) -> Array:
        bd_array = []
        for i, bc in enumerate(self.bc.bd_infos[1:]): 
            # find sid and eid 
            sid = grid.patch_sids[i]
            eid = grid.patch_eids[i]
            temp_func = bc[-1]
            # print('bc',i, temp_func)
            if bc[1] == 'fixedValue':
                bd_array.append(temp_func(grid.face_centers[sid:eid])) 
            elif bc[1] == 'zeroGradient':
                bd_array.append(self.cell_phi[grid.mesh.owners[sid:eid]])
            elif bc[1] == 'empty':
                bd_array.append(self.cell_phi[grid.mesh.owners[sid:eid]])
        # for ele in bd_array:
            # print('bd array shapes: ', ele.shape)
        return jnp.array(jnp.concatenate(bd_array)) # phi values on the boundary 

    def UpdateBoundaryPhi(self,grid: Grid) -> GridVariable: 
        return GridVariable(self.cell_phi, self.GetBoundaryPhi(grid), self.timestep, self.bc, self.name)


GridVariableVector = Tuple[GridVariable, ...]

@dataclasses.dataclass(init=False, frozen=True)
class Grid:
    """
    This is an unstructure grid
    """
    n_nodes: int
    n_cells: int
    dim: int
    n_faces: int
    n_I_faces: int
    n_B_faces: int
    # n_cells_B_Faces: int
    patch_names: Tuple[str, ...]
    patch_sids: Tuple[int, ...]
    patch_n_faces: Tuple[int, ...]
    patch_eids: Tuple[int, ...]

    def __init__(self, region):
        '''
        The input should be a npz file of all the information of a mesh 
        '''
        mesh = region.mesh
        # essentials 
        object.__setattr__(self, 'nodes',jnp.array(mesh.nodeCentroids)) # num of vertices 
        object.__setattr__(self, 'faces',mesh.faceNodes) # num of vertices 
        object.__setattr__(self, 'owners',jnp.array(mesh.owners)) # num of vertices 
        object.__setattr__(self, 'neighbours',jnp.array(mesh.neighbours)) # num of vertices
        patch_names = []
        patch_sids = []
        patch_n_faces = [] 
        for key, value in mesh.cfdBoundaryPatchesArray.items():
            patch_names.append(key)
            patch_sids.append(value['startFaceIndex'])
            patch_n_faces.append(value['numberOfBFaces'])
        object.__setattr__(self, 'patch_names', tuple(patch_names))
        object.__setattr__(self, 'patch_sids', tuple(patch_sids))
        object.__setattr__(self, 'patch_n_faces', tuple(patch_n_faces))
        
        # yongqi comment: add patch_eids to save my life :)
        # Iterate over the indices of the tuples
        patch_eids = []
        for i in range(len(patch_sids)):
            summed = patch_sids[i] + patch_n_faces[i]
            patch_eids.append(summed)
        object.__setattr__(self, "patch_eids", tuple(patch_eids))

        # inferred simple attributes 
        object.__setattr__(self, 'n_nodes',self.nodes.shape[0]) # num of graph nodes 
        object.__setattr__(self, 'dim',self.nodes.shape[1]) # dimension of the mesh
        object.__setattr__(self, 'n_cells',max(mesh.neighbours) + 1) # num of cells #yongqi comment: a bug
        object.__setattr__(self, 'n_faces',len(self.faces))  # number of total faces #yongqi comment: change shape to len
        object.__setattr__(self, 'n_I_faces',self.neighbours.shape[0]) # number of internal faces 
        object.__setattr__(self, 'n_B_faces',self.n_faces-self.n_I_faces) # number of internal faces 

        # extra properties 
        object.__setattr__(self, 'face_centers',jnp.array(mesh.faceCentroids))
        object.__setattr__(self, 'faceSf',jnp.array(mesh.faceSf))# surface vector #yongqi comment: change face_Sf to faceSf
        object.__setattr__(self, 'face_areas',jnp.array(mesh.faceAreas))
        object.__setattr__(self, 'ratio',jnp.array(mesh.faceWeights))
        object.__setattr__(self, 'cell_centers',jnp.array(mesh.elementCentroids)) # sender node
        object.__setattr__(self, 'faceCF',jnp.array(mesh.faceCF)) # 
        object.__setattr__(self, 'faceCf',jnp.array(mesh.faceCf)) # 
        object.__setattr__(self, 'faceFf',jnp.array(mesh.faceFf)) # 
        object.__setattr__(self, 'wallDist',jnp.array(mesh.wallDist)) # 
        object.__setattr__(self, 'wallDistLimited',jnp.array(mesh.wallDistLimited)) # 
        
        # gamma needs to further adjust for intrepolation
        object.__setattr__(self, 'gamma',jnp.array(mesh.gamma))
        
        #add schemes to grids
        object.__setattr__(self, "snGradSchemes", region.foamDictionary['fvSchemes']['snGradSchemes'])
        object.__setattr__(self, "gradSchemes", region.foamDictionary['fvSchemes']['gradSchemes'])
        object.__setattr__(self, "interpolationSchemes", region.foamDictionary['fvSchemes']['interpolationSchemes'])
        object.__setattr__(self, "ddtSchemes", region.foamDictionary['fvSchemes']['ddtSchemes'])
        object.__setattr__(self, "deltaT",  region.foamDictionary['controlDict']['deltaT'])
        
        #yongqi add: these extra parts are for gDiff and snGradSchemes
        object.__setattr__(
            self,
            "edge_index",
            jnp.vstack((self.owners[: self.n_I_faces], self.neighbours)),
        )
        
        object.__setattr__(
            self,
            "face_owner_ud",
            jnp.concatenate(
                (
                    self.edge_index[0],
                    self.edge_index[1],
                    self.owners[self.n_I_faces :],
                ),
                axis=0,
            ),
        )  # including double internal face and bd face
        
        Ef_norm, Tf = ComputeEfnormTf(region, self)
        Cb_dist = jnp.sum(
            self.faceCF[self.n_I_faces :]
            * self.faceSf[self.n_I_faces :]
            / self.face_areas[self.n_I_faces :][..., None],
            axis=1,
        )  # distance between boundary element centeroids and its faces centeroids
        dCF = jnp.concatenate(
            (jnp.linalg.norm((self.faceCF), axis=1)[: self.n_I_faces], Cb_dist)
        )
        object.__setattr__(self, "dCF", dCF)
        object.__setattr__(self, "Ef_norm", Ef_norm)
        object.__setattr__(self, "faceTf", Tf)
        object.__setattr__(self, "gDiff", self.Ef_norm / self.dCF)
        
        scalar_scatter_dim_num = lax.ScatterDimensionNumbers((), (0,), (0,))
        vector_scatter_dim_num = lax.ScatterDimensionNumbers(
            (1,), (0,), (0,)
        )  # map scatter to operand
        object.__setattr__(self, "scalar_scatter_dim_num", scalar_scatter_dim_num)
        object.__setattr__(self, "vector_scatter_dim_num", vector_scatter_dim_num)  
        object.__setattr__(self, "cell_volumes", jnp.array(mesh.elementVolumes))
    
    @property
    def GetNumOfDim(self) -> int:
        """Returns the number of dimensions of this grid."""
        return self.dim

    @property
    def GetNumOfNodes(self) -> int:
        """Returns the number of nodes of this grid."""
        return self.n_nodes

    @property
    def GetNumOfCells(self) -> int:
        """Returns the number of cells of this grid."""
        return self.n_cells

    def EvaluateOnMesh(self, fn: Callable[..., Array], region: str) -> Array:
        """Evaluates the function on the grid mesh."""
        if region == "cell":
            return fn(self.cell_centers)
        elif region == "boundary":
            return fn(self.face_centers[self.n_I_faces :])
        elif region == "cell and boundary":
            C_pos = self.cell_centers[self.edge_index[0]]
            return fn(jnp.vstack(C_pos, self.face_centers[self.n_I_faces :]))
        elif region == "face":
            return fn(self.face_centers)
    
    def SetUpSimulation(self, **kwargs):
        """
        Inputs:
            **kwargs: params needed for the corresponding equation
        """
        for key, value in kwargs.items():
            if key in ["source", "gt"]:
                if type(value) == float or (
                    type(value) == Array and value.shape == (1,)
                ):
                    myfield = jnp.ones(self.n_cells)
                elif type(value) == Array:
                    myfield = value
                elif callable(value):
                    myfield = self.EvaluateOnMesh(value, "cell")
                else:
                    raise Exception("input type is not supported")
                object.__setattr__(self, key, myfield)

            else:
                object.__setattr__(self, key, value)
    # def SetUpSimulation(self, equation_name, **kwargs):
    #     '''
    #     Inputs:
    #         equation_name: name of the equation
    #         **kwargs: params needed for the corresponding equation 
    #     '''
    #     self.Precompute()
    #     object.__setattr__(self, 'equation',equation_name)
    #     if equation_name == 'SteadyDiffusion':
    #         for key, value in kwargs.items(): 
    #             if key in ['source', 'gt']: 
    #                 if type(value) == float or (type(value)==Array and value.shape == (1,)): 
    #                     myfield = jnp.ones(self.n_cells)
    #                 elif type(value) == Array: 
    #                     myfield = value
    #                 elif callable(value):
    #                     myfield = self.EvaluateOnMesh(value, 'cell')
    #                 else: 
    #                     raise Exception("input type is not supported")
    #                 object.__setattr__(self, key, myfield)

    #             elif 'sigma' == key: # we need sigmas on the faces 
    #                 if type(value) == float or (type(value)==Array and value.shape == (1,)): 
    #                     myfield = jnp.ones(self.N_f)
    #                 elif type(value) == Array: 
    #                     myfield = value
    #                 elif callable(value):
    #                     myfield = self.EvaluateOnMesh(value, 'face')
    #                 else: 
    #                     raise Exception("input type is not supported")
    #                 object.__setattr__(self, key, myfield)
    #             else:      
    #                 object.__setattr__(self, key, value)

    #     elif equation_name == 'Convection Diffusion':
    #         #TBI
    #         pass
    #     elif equation_name == 'Navier Stokes':
    #         #TBI
    #         pass

    def GetAllBoundaryValues(self, foi):
        bd_value_list = []
        for i,bd_info in enumerate(foi.bc.bd_infos[1:]): #yongqi comment: skip interior
            bd_value_list.append(self.GetBoundaryValues(i,bd_info,foi))
        return tuple(bd_value_list)
    
    def GetBoundaryValues(self, i, bd_info, foi):
        # assert bd_name in self.patch_names, "The boundary name is not detected" 
        bd_name, bd_info_type, func = bd_info
        sid = self.patch_sids[i]
        eid = self.patch_eids[i]
        
        #yongqi comment: need to be test
        if bd_info_type == 'fixedValue':
            return func(self.face_centers[sid:eid])
        
        
        elif bd_info_type == 'zeroGradient':
            return foi.cell_phi[self.owners[sid:eid]]
        
        elif bd_info_type =='empty':
            return foi.cell_phi[self.owners[sid:eid]]
        
