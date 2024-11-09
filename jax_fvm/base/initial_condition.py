from base import grids
import re
import jax.numpy as jnp 
from typing import Union
from base.region import Region
from utilities.evaluate_functions import CreateFUniformScalar
Array = grids.Array
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector



def InitialField(
    foi_name: str, 
    region: Region,
) -> Union[GridVariable,GridVariableVector]:
    foi_settings = region.fluid[foi_name]
    my_phi_len, my_phi_dim = foi_settings['shape']
    if my_phi_dim ==1: 
        internal_value_type, temp_value_str = foi_settings['internalField'].split()
        if internal_value_type == 'uniform': 
            internal_value = float(temp_value_str)
            internal_func = CreateFUniformScalar(float(internal_value))
        elif internal_value_type == 'specified': 
            # TBD
            pass 
        # next we build up the bc for all the patches. 
        bc_infos = [('interior', 'fixedValue', internal_func)]
        for key, value  in foi_settings['boundaryField'].items(): # will for loop all the patches
            bc_name = key 
            bc_type = value['type']
            if bc_type == 'fixedValue': 
                value_type, value_str = value['value'].split()
                if value_type == 'uniform': 
                    bc_func = CreateFUniformScalar(float(value_str))
            elif bc_type == 'zeroGradient': 
                bc_func = None
            elif bc_type == 'empty':  # this means frontAndBack
                bc_func = None
            bc_infos.append((bc_name, bc_type,bc_func))

        cell_phi = jnp.zeros((my_phi_len[0]))
        bd_phi = jnp.zeros((my_phi_len[1]))
        startTime = float(region.foamDictionary['controlDict']['startTime'])
        my_field = GridVariable(cell_phi, bd_phi, startTime ,grids.BoundaryConditions(bc_infos),foi_name)
        my_field = my_field.InitializeInterior(region) #yongqi comment: Here we need specify my_field again to update the value of my_field
        
        my_field = my_field.UpdateBoundaryPhi(region.mesh)
        
    elif my_phi_dim ==3: # yongqi comment: need to double check this part
        match = re.match(r"(\w+)\s*\((\d+)\s+(\d+)\s+(\d+)\)", foi_settings['internalField'])
        internal_value_type = match.group(1)  # 'uniform'
        temp_value = tuple(float(num) for num in match.groups()[1:])  # (2, 1, 0)
        startTime = float(region.foamDictionary['controlDict']['startTime'])
        # print('hahaha', internal_value_type, temp_value)
        my_field = []
        for i in range(my_phi_dim):
            if internal_value_type == 'uniform': 
                internal_value = temp_value[i]
                internal_func = CreateFUniformScalar(float(internal_value))
            elif internal_value_type == 'specified': 
                # TBD
                pass   
            # next we build up the bc for all the patches. 
            bc_infos = [('interior', 'fixedValue', internal_func)]
            for key, value  in foi_settings['boundaryField'].items(): # will for loop all the patches
                bc_name = key 
                bc_type = value['type']
                if bc_type == 'fixedValue': 
                    value_type = match.group(1)
                    if value_type == 'uniform': 
                        match = re.match(r"(\w+)\s*\((\d+)\s+(\d+)\s+(\d+)\)", value['value'])
                        value_str = match.groups()[i+1]
                        bc_func = CreateFUniformScalar(float(value_str))
                elif bc_type == 'zeroGradient': 
                    bc_func = None
                elif bc_type == 'empty':  # this means frontAndBack
                    bc_func = None
                bc_infos.append((bc_name, bc_type,bc_func))     
            cell_phi = jnp.zeros((my_phi_len[0]))
            bd_phi = jnp.zeros((my_phi_len[1]))
            my_field.append(GridVariable(cell_phi, bd_phi, startTime, grids.BoundaryConditions(bc_infos),foi_name))
            my_field[-1].InitializeInterior(region)
            my_field[-1].UpdateBoundaryPhi(region.mesh)
        my_field = tuple(my_field)
        
    return my_field
