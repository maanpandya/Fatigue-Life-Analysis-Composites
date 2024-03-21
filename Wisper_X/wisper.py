import numpy as np
import rainflow 

# Read the file
with open('Wisper_X/WISPERX', 'r') as file:
    file_contents = file.readlines()

array = np.array([float(x) for x in file_contents[1:]])

# Normalize the array to the maximum value
normalized_array = array / np.max(array)

# Count cycles
counted_cycles = rainflow.count_cycles(normalized_array)


# Print the counted cycles
print(counted_cycles)
print(normalized_array)