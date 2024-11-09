import os 
from base.region import Region

def cfdGetTimeSteps( region:Region ):
    # Get the list of files and folders in the current directory
    items = os.listdir(region.caseDirectoryPath)
    
    # Initialize an empty list to store integer folder names
    int_folders = []
    
    # Loop through the items and check for directories
    for item in items:
        if os.path.isdir(item):
            try:
                # Try to convert the folder name to an integer
                folder_name = int(item)
                # If successful, add it to the list
                int_folders.append(folder_name)
            except ValueError:
                # If conversion fails, it's not an integer-named folder, ignore it
                continue
    
    # Sort the integers in increasing order
    int_folders.sort()
    return int_folders
