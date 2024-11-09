import os
from os.path import join as osj
from base.region import Region
from mesh.IO import (
    cfdReadPointsFile,
    cfdReadFacesFile,
    cfdReadOwnerFile,
    cfdReadNeighbourFile,
    cfdReadBoundaryFile,
)

from timestep.cfdGetTimeSteps import cfdGetTimeSteps
from fields.vector.cfdSetupMeshField import cfdSetupMeshField
from utilities.others import ParseDictionaryFile

import jax.numpy as jnp

def cfdReadControlDictFile(region: Region):
    # class ControlDict:
    #     pass
    # object.__setattr__(region.foamDictionary, 'controlDict', ControlDict())
    region.foamDictionary["controlDict"] = dict()

    file_path = region.caseDirectoryPath + "/system/controlDict"
    myDict = ParseDictionaryFile(file_path)
    # print('debuggin', myDict)
    for key, value in myDict.items():
        if "FoamFile" in key:
            pass
        elif "application" in key:
            region.foamDictionary["controlDict"][key] = value
        elif "startFrom" in key:
            region.foamDictionary["controlDict"][key] = value
        elif "startTime" in key:
            region.foamDictionary["controlDict"][key] = float(value)
        elif "stopAt" in key:
            region.foamDictionary["controlDict"][key] = value
        elif "endTime" in key:
            region.foamDictionary["controlDict"][key] = int(value)
        elif "deltaT" in key:
            region.foamDictionary["controlDict"][key] = float(value) #yongqi comment: change int to float
        elif "writeControl" in key:
            region.foamDictionary["controlDict"][key] = value
        elif "writeInterval" in key:
            region.foamDictionary["controlDict"][key] = int(value)
        elif "purgeWrite" in key:
            region.foamDictionary["controlDict"][key] = int(value)


def cfdReadFvSchemes(region: Region):
    # class FvSchemes:
    #     pass
    # object.__setattr__(region.foamDictionary, 'fvSchemes', FvSchemes())
    region.foamDictionary["fvSchemes"] = dict()
    file_path = region.caseDirectoryPath + "/system/fvSchemes"
    myDict = ParseDictionaryFile(file_path)
    # print(myDict)
    for key, value in myDict.items():
        if "FoamFile" in key:
            pass
        elif "ddtSchemes" in key:
            # print('debug:', key, value['default'])
            region.foamDictionary["fvSchemes"][key] = value["default"]
            # if value['default'] == "Steady":
            #     object.__setattr__(region.foamDictionary.fvSchemes, "Steady", True)
            # else:
            #     object.__setattr__(region.foamDictionary.fvSchemes, "Steady", False)

        elif "gradSchemes" in key:
            # print('debug:', key, value['default'])
            region.foamDictionary["fvSchemes"][key] = value["default"]
            # object.__setattr__(region.foamDictionary.fvSchemes, key, value['default'])
        elif "divSchemes" in key:
            region.foamDictionary["fvSchemes"][key] = value["default"]
            # print('debug:', key, value['default'])
            # object.__setattr__(region.foamDictionary.fvSchemes, key, value['default'])
        elif "laplacianSchemes" in key:
            region.foamDictionary["fvSchemes"][key] = value["default"]
            # print('debug:', key, value['default'])
            # object.__setattr__(region.foamDictionary.fvSchemes, key, value['default'])
        elif "interpolationSchemes" in key:
            region.foamDictionary["fvSchemes"][key] = value["default"]
            # print('debug:', key, value['default'])
            # object.__setattr__(region.foamDictionary.fvSchemes, key, value['default'])
        elif "snGradSchemes" in key:
            region.foamDictionary["fvSchemes"][key] = value["default"]
            # print('debug:', key, value['default'])
            # object.__setattr__(region.foamDictionary.fvSchemes, key, value['default'])


def cfdReadFvSolutions(region: Region):
    region.foamDictionary["fvSolution"] = dict()
    region.foamDictionary["fvSolution"]["solvers"] = dict()
    region.foamDictionary["fvSolution"]["solvers"]["SIMPLE"] = dict()
    region.foamDictionary["fvSolution"]["solvers"]["relaxationFactors"] = dict()
    file_path = region.caseDirectoryPath + "/system/fvSolution"
    myDict = ParseDictionaryFile(file_path)
    # print(myDict)
    for key, value in myDict.items():
        if "FoamFile" in key:
            pass
        elif "solvers" in key:
            for key1, value1 in myDict[key].items():
                region.foamDictionary["fvSolution"]["solvers"][key1] = dict()
                for key2, value2 in myDict[key][key1].items():
                    if (
                        "solver" in key2
                        or "preconditioner" in key2
                        or "smoother" in key2
                    ):
                        region.foamDictionary["fvSolution"]["solvers"][key1][
                            key2
                        ] = value2
                    else:
                        region.foamDictionary["fvSolution"]["solvers"][key1][key2] = (
                            float(value2)
                        )
                        # object.__setattr__(getattr(region.foamDictionary.fvSolution.solvers, key1),key2, float(value2))
                if not "maxIter" in list(myDict[key][key1].keys()):
                    region.foamDictionary["fvSolution"]["solvers"][key1]["maxIter"] = 20
                    # object.__setattr__(getattr(region.foamDictionary.fvSolution.solvers, key1),"maxIter", 20)

                if (
                    region.foamDictionary["fvSolution"]["solvers"][key1]["solver"]
                    == "GAMG"
                ):
                    if not "nPreSweeps" in list(myDict[key][key1].keys()):
                        region.foamDictionary["fvSolution"]["solvers"][key1][
                            "nPreSweeps"
                        ] = 0
                        # object.__setattr__(getattr(region.foamDictionary.fvSolution.solvers, key1),"nPreSweeps", 0)
                    if not "nPostSweeps" in list(myDict[key][key1].keys()):
                        region.foamDictionary["fvSolution"]["solvers"][key1][
                            "nPostSweeps"
                        ] = 2
                        # object.__setattr__(getattr(region.foamDictionary.fvSolution.solvers, key1),"nPostSweeps", 2)
                    if not "nFinestSweeps" in list(myDict[key][key1].keys()):
                        region.foamDictionary["fvSolution"]["solvers"][key1][
                            "nFinestSweeps"
                        ] = 2
                        # object.__setattr__(getattr(region.foamDictionary.fvSolution.solvers, key1),"nFinestSweeps", 2)
        elif "SIMPLE" in key:
            for key1, value1 in myDict[key].items():
                # CreateSubDummayClass(region.foamDictionary.solvers, key1)
                if "pRefCell" in key1:
                    region.foamDictionary["fvSolution"]["solvers"]["SIMPLE"][key1] = (
                        int(value1) + 1
                    )
                    # object.__setattr__(region.foamDictionary.fvSolution.SIMPLE, key1, int(value1)+1)
                elif "residualControl" in key1:
                    region.foamDictionary["fvSolution"]["solvers"]["SIMPLE"][
                        key1
                    ] = dict()
                    # CreateSubDummayClass(region.foamDictionary.fvSolution.SIMPLE, key1)
                    for fieldName in list(myDict["solvers"].keys()):
                        if fieldName in list(myDict[key][key1].keys()):
                            region.foamDictionary["fvSolution"]["solvers"]["SIMPLE"][
                                key1
                            ][fieldName] = float(myDict[key][key1][fieldName])
                            # object.__setattr__(getattr(region.foamDictionary.fvSolution.SIMPLE, key1),fieldName, float(myDict[key][key1][fieldName]))
                        else:
                            region.foamDictionary["fvSolution"]["solvers"]["SIMPLE"][
                                key1
                            ][fieldName] = 1e-6
                            # object.__setattr__(getattr(region.foamDictionary.fvSolution.SIMPLE, key1),fieldName, 1e-6)
                else:
                    region.foamDictionary["fvSolution"]["solvers"]["SIMPLE"][key1] = (
                        int(value1)
                    )
                    # object.__setattr__(region.foamDictionary.fvSolution, key1.SIMPLE, int(values))
            if not "pRefCell" in list(myDict[key].keys()):
                region.foamDictionary["fvSolution"]["solvers"]["SIMPLE"]["pRefCell"] = 1
                # object.__setattr__(region.foamDictionary.fvSolution.SIMPLE, "pRefCell", 1)
            if not "pRefValue" in list(myDict[key].keys()):
                region.foamDictionary["fvSolution"]["solvers"]["SIMPLE"][
                    "pRefValue"
                ] = 0
                # object.__setattr__(region.foamDictionary.fvSolution.SIMPLE, "pRefValue", 0)
        elif "relaxationFactors" in key:
            for key1, value1 in myDict[key].items():
                if "equations" in key1:
                    region.foamDictionary["fvSolution"]["solvers"]["relaxationFactors"][
                        key1
                    ] = dict()
                    for fieldName in list(myDict["solvers"].keys()):
                        if fieldName in list(myDict[key][key1].keys()):
                            region.foamDictionary["fvSolution"]["solvers"][
                                "relaxationFactors"
                            ][key1][fieldName] = float(myDict[key][key1][fieldName])
                            # object.__setattr__(getattr(region.foamDictionary.fvSolution.relaxationFactors, key1),fieldName, float(myDict[key][key1][fieldName]))
                        else:
                            region.foamDictionary["fvSolution"]["solvers"][
                                "relaxationFactors"
                            ][key1][fieldName] = 1.0
                            # object.__setattr__(getattr(region.foamDictionary.fvSolution.relaxationFactors, key1),fieldName, 1.0)
                elif "fields" in key1:
                    region.foamDictionary["fvSolution"]["solvers"]["relaxationFactors"][
                        key1
                    ] = dict()
                    for fieldName in list(myDict["solvers"].keys()):
                        if fieldName in list(myDict[key][key1].keys()):
                            region.foamDictionary["fvSolution"]["solvers"][
                                "relaxationFactors"
                            ][key1][fieldName] = float(myDict[key][key1][fieldName])
                            # object.__setattr__(getattr(region.foamDictionary.fvSolution.relaxationFactors, key1),fieldName, float(myDict[key][key1][fieldName]))
                        else:
                            region.foamDictionary["fvSolution"]["solvers"][
                                "relaxationFactors"
                            ][key1][fieldName] = 1.0
                            # object.__setattr__(getattr(region.foamDictionary.fvSolution.relaxationFactors, key1),fieldName, 1.0)

def cfdReadTransportProperties(region: Region):
    file_path = region.caseDirectoryPath + "/constant/transportProperties"
    myDict = ParseDictionaryFile(file_path)
    if "DT" in myDict.keys():
        value = myDict["DT"].split()[-1]
        value = float(value)
        object.__setattr__(region.mesh, "gamma", value * jnp.ones(len(region.mesh.faceNodes)))

def cfdReadTimeDirectory(region: Region):
    if region.foamDictionary["controlDict"]["startFrom"] == "startTime":
        timeDirectory = str(
            "{:.0f}".format(region.foamDictionary["controlDict"]["startTime"])
        )
    elif region.foamDictionary["controlDict"]["startFrom"] == "firstTime":
        timeDirectory = "0"
    elif region.foamDictionary["controlDict"]["startFrom"] == "latestTime":
        timeDirectories = cfdGetTimeSteps()
        timeDirectory = str(max(timeDirectories))
    # read in the fields
    files = os.listdir(
        osj(region.caseDirectoryPath, timeDirectory)
    )  # this will store the fn for all the fields.

    theMesh = region.mesh
    theNumberOfElements = region.mesh.n_cells
    theNumberOfInteriorFaces = region.mesh.n_I_faces
    fois = []
    for file in files:
        fileFullPath = osj(region.caseDirectoryPath, timeDirectory, file)

        if os.path.isdir(fileFullPath) or os.path.getsize(fileFullPath) == 0:
            continue

        # Get field name from file name
        fieldName = file

        myDict = ParseDictionaryFile(fileFullPath)
        print(myDict)

        # set up the fields according to the myDict
        fois.append(cfdSetupMeshField(myDict, region))

        # read in the header
        # myDict = ParseDictionaryFile()
        # print(myDict)



def cfdReadPolyMesh(region: Region):
    cfdReadPointsFile(region)
    cfdReadFacesFile(region)
    cfdReadOwnerFile(region)
    cfdReadNeighbourFile(region)
    cfdReadBoundaryFile(region)
