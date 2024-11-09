from base.grids import Grid
from base.region import Region
import numpy as np 
from utilities.IO.Foam import cfdReadTransportProperties

def cfdProcessElementTopology(region:Region):
    # Info
    theNumberOfElements = max([region.mesh.owners.max(),region.mesh.neighbours.max()])+1
    theNumberOfInteriorFaces = region.mesh.neighbours.shape[0]
    theNumberOfFaces = len(region.mesh.faceNodes)
    owners = region.mesh.owners
    neighbours = region.mesh.neighbours
    faceNodes = region.mesh.faceNodes
    
    # Element-neighbours indices, Element-faces indices
    elementNeighbours = [[] for _ in range(theNumberOfElements)]
    elementFaces = [[] for _ in range(theNumberOfElements)]
    
    # print('debug', theNumberOfElements, theNumberOfInteriorFaces)
    # Interior Faces
    for iFace in range(theNumberOfInteriorFaces):
        own = owners[iFace]
        nei = neighbours[iFace]
        # print(own, nei)
        elementNeighbours[own].append(nei)
        elementNeighbours[nei].append(own)
        
        elementFaces[own].append(iFace)
        elementFaces[nei].append(iFace)
    
    # Boundary patches
    for iFace in range(theNumberOfInteriorFaces, theNumberOfFaces):
        own = owners[iFace]
        elementFaces[own].append(iFace)
    
    # Element-node indices
    elementNodes = [[] for _ in range(theNumberOfElements)]
    for iElement in range(theNumberOfElements):
        for faceIndex in elementFaces[iElement]:
            elementNodes[iElement].extend(faceNodes[faceIndex])
        
        # Remove repetitions in node indices
        elementNodes[iElement] = list(np.unique(elementNodes[iElement]))

    object.__setattr__(region.mesh, 'elementNeighbours',elementNeighbours)
    object.__setattr__(region.mesh, 'elementFaces',elementFaces)
    object.__setattr__(region.mesh, 'elementNodes',elementNodes)

def cfdInvertConnectivity(the_connectivity_array):
    
    # Find the maximum size for the inverted array
    inverted_size = 0
    for i in range(len(the_connectivity_array)):
        for j in the_connectivity_array[i]:
            inverted_size = max(inverted_size, j)
    inverted_size+=1
    # Initialize the inverted connectivity array as a list of empty lists
    the_inverted_connectivity_array = [[] for _ in range(inverted_size)]
    
    # Invert the connectivity
    for i in range(len(the_connectivity_array)):
        for j in the_connectivity_array[i]:
            the_inverted_connectivity_array[j].append(i)  # Adjust for 0-based indexing in Python
    return the_inverted_connectivity_array

def cfdProcessNodeTopology(region:Region):
    
    # Get element and face node indices
    elementNodes = region.mesh.elementNodes
    faceNodes = region.mesh.faceNodes
    
    # Invert connectivity
    nodeElements = cfdInvertConnectivity(elementNodes)
    nodeFaces = cfdInvertConnectivity(faceNodes)
    
    object.__setattr__(region.mesh, 'nodeElements',nodeElements)
    object.__setattr__(region.mesh, 'nodeFaces',nodeFaces)

def cfdUnit(vector):
    return vector / np.linalg.norm(vector)

def cfdProcessGeometry(region: Region):
    """
    This function processes the geometry of the mesh and stores it in the database.
    """
    # Get info
    theNumberOfElements = max([region.mesh.owners.max(),region.mesh.neighbours.max()])+1
    theNumberOfFaces = len(region.mesh.faceNodes)
    theNumberOfInteriorFaces = region.mesh.neighbours.shape[0]
    theFaceNodesIndices = region.mesh.faceNodes
    theNodeCentroids = region.mesh.nodeCentroids
    theElementFaceIndices = region.mesh.elementFaces
    owners = region.mesh.owners
    neighbours = region.mesh.neighbours
    
    # Initialize mesh member arrays
    elementCentroids = np.zeros((theNumberOfElements, 3))
    elementVolumes = np.zeros(theNumberOfElements)
    faceCentroids = np.zeros((theNumberOfFaces, 3))
    faceSf = np.zeros((theNumberOfFaces, 3))
    faceAreas = np.zeros(theNumberOfFaces)
    faceWeights = np.zeros(theNumberOfFaces)
    faceCF = np.zeros((theNumberOfFaces, 3))
    faceCf = np.zeros((theNumberOfFaces, 3))
    faceFf = np.zeros((theNumberOfFaces, 3))
    wallDist = np.zeros(theNumberOfFaces)
    wallDistLimited = np.zeros(theNumberOfFaces)
    
    # Process basic face geometry
    for iFace in range(theNumberOfFaces):
        theNodesIndices = theFaceNodesIndices[iFace]
        theNumberOfFaceNodes = len(theNodesIndices)
        
        # Compute a rough center of the face
        local_centre = np.zeros(3)
        for iNode in theNodesIndices:
            local_centre += theNodeCentroids[iNode]
        local_centre /= theNumberOfFaceNodes
        
        centroid = np.zeros(3)
        Sf = np.zeros(3)
        area = 0
        
        # Compute the area and centroid based on virtual triangles
        for iTriangle in range(theNumberOfFaceNodes):
            point1 = local_centre
            point2 = theNodeCentroids[theNodesIndices[iTriangle]]
            point3 = theNodeCentroids[theNodesIndices[(iTriangle + 1) % theNumberOfFaceNodes]]
            
            local_centroid = (point1 + point2 + point3) / 3
            local_Sf = 0.5 * np.cross(point2 - point1, point3 - point1)
            local_area = np.linalg.norm(local_Sf)
            
            centroid += local_area * local_centroid
            Sf += local_Sf
            area += local_area
        
        centroid /= area
        faceCentroids[iFace] = centroid
        faceSf[iFace] = Sf
        faceAreas[iFace] = area
    
    # Compute volume and centroid of each element
    for iElement in range(theNumberOfElements):
        theElementFaces = theElementFaceIndices[iElement]
        
        # Compute a rough center of the element
        local_centre = np.zeros(3)
        for iFace in theElementFaces:
            local_centre += faceCentroids[iFace]
        local_centre /= len(theElementFaces)
        
        localVolumeCentroidSum = np.zeros(3)
        localVolumeSum = 0
        
        for iFace in theElementFaces:
            Cf = faceCentroids[iFace] - local_centre
            
            faceSign = 1 if iElement == owners[iFace] else -1
            local_Sf = faceSign * faceSf[iFace]
            
            localVolume = np.dot(local_Sf, Cf) / 3
            localCentroid = 0.75 * faceCentroids[iFace] + 0.25 * local_centre
            
            localVolumeCentroidSum += localCentroid * localVolume
            localVolumeSum += localVolume
        
        elementCentroids[iElement] = localVolumeCentroidSum / localVolumeSum
        elementVolumes[iElement] = localVolumeSum
    
    # Process secondary face geometry
    for iFace in range(theNumberOfInteriorFaces):
        n = cfdUnit(faceSf[iFace])
        own = owners[iFace]
        nei = neighbours[iFace]
        
        faceCF[iFace] = elementCentroids[nei] - elementCentroids[own]
        faceCf[iFace] = faceCentroids[iFace] - elementCentroids[own]
        faceFf[iFace] = faceCentroids[iFace] - elementCentroids[nei]
        faceWeights[iFace] = np.dot(faceCf[iFace], n) / (np.dot(faceCf[iFace], n) - np.dot(faceFf[iFace], n))
    
    for iBFace in range(theNumberOfInteriorFaces, theNumberOfFaces):
        n = cfdUnit(faceSf[iBFace])
        own = owners[iBFace]
        
        faceCF[iBFace] = faceCentroids[iBFace] - elementCentroids[own]
        faceCf[iBFace] = faceCentroids[iBFace] - elementCentroids[own]
        faceWeights[iBFace] = 1
        wallDist[iBFace] = max(np.dot(faceCf[iBFace], n), 1e-24)
        wallDistLimited[iBFace] = max(wallDist[iBFace], 0.05 * np.linalg.norm(faceCf[iBFace]))

    object.__setattr__(region.mesh, 'elementCentroids',elementCentroids)
    object.__setattr__(region.mesh, 'elementVolumes',elementVolumes)
    object.__setattr__(region.mesh, 'faceCentroids',faceCentroids)
    object.__setattr__(region.mesh, 'faceSf',faceSf)
    object.__setattr__(region.mesh, 'faceAreas',faceAreas)
    object.__setattr__(region.mesh, 'faceWeights',faceWeights)
    object.__setattr__(region.mesh, 'faceCF',faceCF)
    object.__setattr__(region.mesh, 'faceCf',faceCf)
    object.__setattr__(region.mesh, 'faceFf',faceFf)
    object.__setattr__(region.mesh, 'wallDist',wallDist)
    object.__setattr__(region.mesh, 'wallDistLimited',wallDistLimited)


def cfdConvertToGrid(region: Region):
    print("Converting mesh to jax mesh ...")
    
    grid = Grid(region) #yongqi comment: I replace region.mesh to region because I need foamSchemes from region
    del region.mesh
    object.__setattr__(region, "mesh", grid)
    
def cfdProcessTopology(region: Region):
    print("processing mesh ...")
    cfdProcessElementTopology(region)
    cfdProcessNodeTopology(region)
    cfdProcessGeometry(region)
    cfdReadTransportProperties(region)
    cfdConvertToGrid(region)