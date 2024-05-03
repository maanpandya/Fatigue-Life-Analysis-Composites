import sys
sys.path.append('NeuralNetworkCode')

import numpy as np
import rainflow 
import matplotlib.pyplot as plt
import function as f
import torch
import pandas as pd
from   NNfuncion import NNfuntion 
from   SNfuncion import SNfunction 


# Read the file
with open('NeuralNetworkCode\Var-Amplitude-Test\WISPER', 'r') as file:
    file_contents = file.readlines()
array = np.array([float(x) for x in file_contents[1:]])

# Normalize the array to the maximum value
array = array-25
normalized_array = array/ np.max(array)
counted_cycles = rainflow.count_cycles(normalized_array)
cyclesSN, stressesSN = SNfunction(normalized_array, list(range(10)))
cyclesNN, stressesNN = NNfuntion(normalized_array, list(range(10)))

anotherone = [46.63400685, 30.53759847, 60.13090802 , 86.38254409, 30.11657662, 36.04822509, 25.01408323, 24.02818907, 42.03037427, 2.819182283, 8.159798359, 3.860663396 , 65.88812532 ,11.30806298, 11.0245947, 10.94091318, 69.13812292, 83.13420841 ]
j = [378.90 , 371.15, 379.23, 355.52, 381.00, 365.03, 374.66, 376.13, 349.57, 469.89, 466.83, 470.06, 342.82, 443.18, 444.34, 446.66, 357.72, 356.78]

plt.plot(np.log(cyclesSN), stressesSN, color = "black", label='SN curves')
plt.plot(np.log(cyclesNN), stressesNN, color = "red", label='PINN')
plt.scatter(np.log(anotherone), j, color = "blue", label='Actual values')
plt.ylabel("Stress[MPA]", fontsize = 12)
plt.xlabel("Number of cycles", fontsize = 12)
plt.legend()
plt.show()

cyclesSN, stressesSN = SNfunction(normalized_array, j)
cyclesNN, stressesNN = NNfuntion(normalized_array, j)
print("stressesSN")
print(cyclesSN)
print("stressesNN")
print(cyclesNN)
plt.scatter(np.log(anotherone), np.log(cyclesSN), color = "black")
plt.scatter(np.log(anotherone), np.log(cyclesNN), color = "blue")
plt.xlabel("Logarithm scale number of cycles Optidat", fontsize = 12)
plt.ylabel("Logarithm scale number of cycles Wisper-PINN", fontsize = 12)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
x_values = np.linspace(-10, 10, 200)  # Creating an array of x values from 0 to 5
y_values = x_values  # Since x = y, y values will be the same as x values
plt.plot(x_values, y_values, color='red', linestyle='--')
plt.show()