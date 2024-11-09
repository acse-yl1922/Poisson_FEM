import re 

def SetKey(myDict, keyList):
    temp = myDict
    for key in keyList[:-1]:
        temp = temp[key]
    temp[keyList[-1]] = {}

def SetValue(myDict, keyList, pair):
    temp = myDict
    for key in keyList:
        temp = temp[key]
    temp[pair[0]] = pair[1]

def ParseDictionaryFile(file_path):
    # Dictionary to hold the parsed data
    parsed_data = {}

    # Regular expressions to match the block and key-value pairs
    latest_block_pattern = re.compile(r"^\s*(\w+)\s*$")
    key_value_pattern = re.compile(r"\s*(\w+)\s+(.+);")
    
    current_block = None
    inside_foam_file = False

    isHeader = True
    current_scope_lvl = 0
    temp_keyList = [] 
    with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            temp_block_match = latest_block_pattern.match(line)
            if temp_block_match!=None:
                block_match = temp_block_match
                if "FoamFile" in block_match.group(1): 
                    isHeader = False
            # print("block match!", block_match.group(1))
            if isHeader or not line or line.startswith("//") or line.startswith("/*") or line.startswith("*"):
                continue
            
            if "{" in line: 
                # print("next scope, adding",block_match.group(1))
                current_scope_lvl +=1
                temp_keyList.append(block_match.group(1))
                # print(temp_keyList)
                SetKey(parsed_data, temp_keyList)
            elif "}" in line:
                current_scope_lvl -=1
                temp_keyList.pop()

            key_match = key_value_pattern.match(line)
            if key_match != None:
                # print("filling key match")
                SetValue(parsed_data, temp_keyList, (key_match.group(1),key_match.group(2)))

    return parsed_data

def CreateSubDummayClass(myClass, subClassName: str):
    class Dummy():
        pass
    object.__setattr__(myClass, subClassName, Dummy())