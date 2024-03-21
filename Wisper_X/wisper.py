import numpy as np
import rainflow 

# Read the file
with open('Wisper_X/WISPERX', 'r') as file:
    file_contents = file.readlines()

array = np.array([float(x) for x in file_contents[1:]])


# Print the counted cycles
print(counted_cycles)
print(array)