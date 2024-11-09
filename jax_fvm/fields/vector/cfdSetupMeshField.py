from base.region import Region

def cfdSetupMeshField(myDict, region: Region):
    '''
    adding field attributes to myregion class 
    '''
    region.fluid[myDict['FoamFile']['object']]= dict()
    region.fluid[myDict['FoamFile']['object']]['theType'] = myDict['FoamFile']['class']
    # object.__setattr__(getattr(region.fluid, myDict['FoamFile']['object']), 'theType', myDict['FoamFile']['class'])
    if myDict['FoamFile']['class'] == 'volScalarField':
        my_phi_len= (region.mesh.n_cells, region.mesh.n_B_faces)
        my_phi_dim= 1
    elif myDict['FoamFile']['class'] == 'volVectorField':
        my_phi_len = (region.mesh.n_cells, region.mesh.n_B_faces)
        my_phi_dim= 3
    elif myDict['FoamFile']['class'] == 'surfaceScalarField':
        my_phi_len = (region.mesh.n_I_faces, region.mesh.n_B_faces)
        my_phi_dim= 1
    elif myDict['FoamFile']['class'] == 'surfaceVectorField':
        my_phi_len = (region.mesh.n_I_faces, region.mesh.n_B_faces)
        my_phi_dim= 3
    
    region.fluid[myDict['FoamFile']['object']]['shape'] = (my_phi_len, my_phi_dim)
    region.fluid[myDict['FoamFile']['object']]['dimensions'] = myDict['dimensions']
    region.fluid[myDict['FoamFile']['object']]['internalField'] = myDict['internalField']
    region.fluid[myDict['FoamFile']['object']]['boundaryField'] = myDict['boundaryField']
    # object.__setattr__(getattr(region.fluid, myDict['FoamFile']['object']), 'shape', (my_phi_len, my_phi_dim))
    # object.__setattr__(getattr(region.fluid, myDict['FoamFile']['object']), 'dimensions', myDict['dimensions'])
    # object.__setattr__(getattr(region.fluid, myDict['FoamFile']['object']), 'internalField', myDict['internalField'])
    # object.__setattr__(getattr(region.fluid, myDict['FoamFile']['object']), 'boundaryField', myDict['boundaryField'])









