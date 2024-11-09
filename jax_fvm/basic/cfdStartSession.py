import os
from base.region import Region


def cfdPrintMainHeader():
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-* GraphFVM *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
    print("|| Written by Pan Du from University of NotreDame.   ||\n")
    print("|| An academic CFD package developed for learning purposes to serve ||\n")
    print("|| the student community.                                           ||\n")
    print("----------------------------------------------------------------------\n")
    print(" Credits:\n \tPan Du, Jian-Xun Wang\n")
    print("\pdu@nd.edu\n")
    print("\tUniversity Of Notre Dame\n")
    print("\tGraphFVM v1.0, 2024\n")
    print(
        "\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n\n"
    )

#yongqi comment: Here I use user specified path for IO
def cfdSetupRegion(path):
    regionInstance = Region()
    object.__setattr__(
        regionInstance, "caseDirectoryPath", path
    )  # case directory
    object.__setattr__(regionInstance, "STEADY_STATE_RUN", True)  # case directory
    return regionInstance


def cfdPrintCaseDirectoyPath(region: Region):
    print("Simulation Directory: ", region.caseDirectoryPath)
