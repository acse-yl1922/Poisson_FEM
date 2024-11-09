from base.region import Region
from utilities.others import CreateSubDummayClass #yongqi comment: all import of CreateSubDummayClass have been corrected


def cfdSetupMesh(region: Region):
    CreateSubDummayClass(region, "mesh")
