from base.region import Region
import numpy as np
import re
import jax.numpy as jnp

def cfdSkipEmptyLines(tline):
    if not tline.strip():
        tline = False
    else:
        tline = tline
    return tline


def cfdSkipMacroComments(tline):
    trimmedTline = tline.strip()

    if "/*" in trimmedTline:
        tline = False
    elif "|" in trimmedTline:
        tline = False
    elif "\*" in trimmedTline:
        tline = False
    elif "*" in trimmedTline:
        tline = False
    else:
        tline = tline
    return tline


def cfdReadCfdDictionary(fpid, **kwargs):
    subDictionary = False
    dictionary = {}

    if "line" in kwargs:
        dictionary[kwargs.get("line")[0]] = kwargs.get("line")[1]

    for _, tline in enumerate(fpid):

        if not cfdSkipEmptyLines(tline):
            continue

        if not cfdSkipMacroComments(tline):
            continue

        if "{" in tline:
            continue

        # check for end of subDictionary
        if "}" in tline and subDictionary == True:
            subDictionary = False
            continue

        if "}" in tline and subDictionary == False:
            break

        if len(tline.split()) == 1 and "$" in tline and subDictionary == True:
            DictKey = tline.replace("$", "").replace("\n", "").replace(";", "").strip()
            dictionary[currentSubDictKey] = {
                **dictionary[currentSubDictKey],
                **dictionary[DictKey],
            }
            # for key in dictionary[DictKey]:
            #     dictionary[currentSubDictKey].setdefault(key, dictionary[DictKey][key])
            continue

        tline = tline.replace(";", "")

        if len(tline.split()) == 1 and subDictionary == False:

            subDictionary = True
            dictionary[tline.split()[0]] = {}
            currentSubDictKey = tline.split()[0]
            continue

        if subDictionary == True:
            try:
                dictionary[currentSubDictKey][tline.split()[0]] = float(
                    tline.split()[1]
                )

            except ValueError:
                lenth = len(tline.split())
                dictionary[currentSubDictKey][tline.split()[0]] = " ".join(
                    tline.split()[1:lenth]
                )
            continue
        else:
            try:
                dictionary[tline.split()[0]] = float(tline.split()[1])
            except ValueError:
                lenth = len(tline.split())
                dictionary[tline.split()[0]] = " ".join(tline.split()[1:lenth])
    return dictionary


def cfdReadPointsFile(region: Region):
    pointsFile = r"%s/constant/polyMesh/points" % region.caseDirectoryPath
    with open(pointsFile, "r") as fpid:

        print("Reading points file ...")
        points_x = []
        points_y = []
        points_z = []

        for _, tline in enumerate(fpid):

            if not cfdSkipEmptyLines(tline):
                continue

            if not cfdSkipMacroComments(tline):
                continue

            if "FoamFile" in tline:
                dictionary = cfdReadCfdDictionary(fpid)
                continue

            if len(tline.split()) == 1:
                if "(" in tline:
                    continue
                if ")" in tline:
                    continue
                else:
                    continue

            tline = tline.replace("(", "")
            tline = tline.replace(")", "")
            tline = tline.split()

            points_x.append(np.float64(tline[0]))
            points_y.append(np.float64(tline[1]))
            points_z.append(np.float64(tline[2]))

    ## (array) with the mesh point coordinates
    pos = np.array((points_x, points_y, points_z)).transpose()
    object.__setattr__(region.mesh, "nodeCentroids", pos)
    object.__setattr__(region.mesh, "numberOfNodes", pos.shape[0])


def cfdReadFacesFile(region: Region):
    ## Path to faces file
    facesFile = r"%s/constant/polyMesh/faces" % region.caseDirectoryPath
    with open(facesFile, "r") as fpid:
        print("Reading faces file ...")
        faceNodes = []

        for _, tline in enumerate(fpid):

            if not cfdSkipEmptyLines(tline):
                continue

            if not cfdSkipMacroComments(tline):
                continue

            if "FoamFile" in tline:
                dictionary = cfdReadCfdDictionary(fpid)
                continue

            if len(tline.split()) == 1:
                if "(" in tline:
                    continue
                if ")" in tline:
                    continue
                else:

                    numberOfFaces = int(tline.split()[0])
                    continue

            tline = tline.replace("(", " ")
            tline = tline.replace(")", "")
            faceNodesi = []
            for count, node in enumerate(tline.split()):
                if count == 0:
                    continue

                else:
                    faceNodesi.append(int(node))
            faceNodes.append(np.array(faceNodesi)) #yongqi comment: in step profile case faces have different shapes so we will store faces as list of np arrays

    object.__setattr__(region.mesh, "faceNodes", faceNodes)
    object.__setattr__(region.mesh, "numberOfFaces", numberOfFaces)


def cfdReadOwnerFile(region: Region):
    """Reads the polyMesh/constant/owner file and returns a list
    where the indexes are the faces and the corresponding element value is the owner cell
    """
    ## Path to owner file
    ownerFile = r"%s/constant/polyMesh/owner" % region.caseDirectoryPath
    with open(ownerFile, "r") as fpid:
        print("Reading owner file ...")

        ## (list) 1D, indices refer to faces, list value is the face's owner cell
        owners = []
        start = False

        for _, tline in enumerate(fpid):

            if not cfdSkipEmptyLines(tline):
                continue

            if not cfdSkipMacroComments(tline):
                continue

            if "FoamFile" in tline:
                dictionary = cfdReadCfdDictionary(fpid)
                continue

            if len(tline.split()) == 1:

                # load and skip number of owners
                if not start:
                    nbrOwner = tline
                    start = True
                    continue

                if "(" in tline:
                    continue
                if ")" in tline:
                    break
                else:
                    owners.append(int(tline.split()[0]))
    object.__setattr__(region.mesh, "owners", np.array(owners))
    # object.__setattr__(region.mesh, "numberOfFaces", len(owners))


def cfdReadNeighbourFile(region: Region):
    ## Path to neighbour file
    neighbourFile = r"%s/constant/polyMesh/neighbour" % region.caseDirectoryPath

    with open(neighbourFile, "r") as fpid:
        print("Reading neighbour file ...")

        ## (list) 1D, indices refer to faces, list value is the face's neighbour cell
        neighbours = []
        start = False

        for _, tline in enumerate(fpid):

            if not cfdSkipEmptyLines(tline):
                continue

            if not cfdSkipMacroComments(tline):
                continue

            if "FoamFile" in tline:
                dictionary = cfdReadCfdDictionary(fpid)
                continue

            if len(tline.split()) == 1:

                # load and skip number of owners
                if not start:
                    N_if = int(tline)
                    start = True
                    continue

                if "(" in tline:
                    continue
                if ")" in tline:
                    break
                else:
                    neighbours.append(int(tline.split()[0]))

    object.__setattr__(region.mesh, "neighbours", np.array(neighbours))
    object.__setattr__(region.mesh, "numberOfInteriorFaces", N_if)
    object.__setattr__(region.mesh, "numberOfBFaces", region.mesh.numberOfFaces - N_if)


def cfdReadBoundaryFile(region: Region):

    boundaryFile = r"%s/constant/polyMesh/boundary" % region.caseDirectoryPath
    with open(boundaryFile, "r") as fpid:
        print("Reading boundary file ...")

        ## (dict) key for each boundary patch
        cfdBoundaryPatchesArray = {}
        for _, tline in enumerate(fpid):

            if not cfdSkipEmptyLines(tline):
                continue

            if not cfdSkipMacroComments(tline):
                continue

            if "FoamFile" in tline:
                dictionary = cfdReadCfdDictionary(fpid)
                continue

            count = 0
            if len(tline.split()) == 1:
                if "(" in tline:
                    continue
                if ")" in tline:
                    continue

                if tline.strip().isdigit():
                    numberOfBoundaryPatches = tline.split()[0]
                    continue

                boundaryName = tline.split()[0]

                cfdBoundaryPatchesArray[boundaryName] = cfdReadCfdDictionary(fpid)
                ## number of faces for the boundary patch
                cfdBoundaryPatchesArray[boundaryName]["numberOfBFaces"] = int(
                    cfdBoundaryPatchesArray[boundaryName].pop("nFaces")
                )

                ## start face index of the boundary patch in the self.faceNodes
                cfdBoundaryPatchesArray[boundaryName]["startFaceIndex"] = int(
                    cfdBoundaryPatchesArray[boundaryName].pop("startFace")
                )
                count = count + 1

                ## index for boundary face, used for reference
                cfdBoundaryPatchesArray[boundaryName]["index"] = count

    object.__setattr__(region.mesh, "cfdBoundaryPatchesArray", cfdBoundaryPatchesArray)
    object.__setattr__(region.mesh, "numberOfBoundaryPatches", count)


def read_openfoam_filed(file_path="openfoam_cases/poisson_problem2/10/T"):
    # Read the entire file content
    with open(file_path, "r") as f:
        content = f.read()

    # Extract the size of the scalar list using regex
    list_size_match = re.search(r"nonuniform List<scalar>\s+(\d+)", content)
    if list_size_match:
        list_size = int(list_size_match.group(1))
    else:
        raise ValueError("Couldn't find the size of the scalar list.")

    # Extract the scalar values using regex
    scalar_values_match = re.search(r"\(\n([\s\S]*?)\)", content)
    if scalar_values_match:
        scalar_values = scalar_values_match.group(1).strip().split()
        scalar_values = [float(value) for value in scalar_values]
    else:
        raise ValueError("Couldn't find the scalar values.")

    # Convert the list to a numpy array
    scalar_array = jnp.array(scalar_values)

    # Ensure the size matches
    if len(scalar_array) != list_size:
        raise ValueError(
            "The size of the scalar list does not match the expected size."
        )

    return scalar_array


def read_openfoam_linearSystem(file_path):
    # Initialize an empty list to store the numerical values
    data = []

    # Open the file and read its content
    with open(file_path, "r") as file:
        lines = file.readlines()

        # Find the line that contains the number of values (e.g., "9555")
        start_index = 0
        for i, line in enumerate(lines):
            if re.match(r"^\d+$", line.strip()):
                start_index = i + 1
                break

        # Extract the numerical values that are inside the parentheses
        values_section = "".join(lines[start_index:])
        data_strings = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", values_section)
        data = [float(value) for value in data_strings]
    return jnp.array(data)
