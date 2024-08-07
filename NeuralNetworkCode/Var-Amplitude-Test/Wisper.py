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

import CLD_interpolator
from CLD_interpolator import CLD_interpolator_log
surface = CLD_interpolator.surface


# Read the file
with open('NeuralNetworkCode\Var-Amplitude-Test\WISPER', 'r') as file:
    file_contents = file.readlines()
array = np.array([float(x) for x in file_contents[1:]])

# Normalize the array to the maximum value
array = array-25
normalized_array = array/ np.max(array)
counted_cycles = rainflow.count_cycles(normalized_array)
cyclesSN, stressesSN = SNfunction(normalized_array, list(range(180, 370)), surface)
cyclesNN, stressesNN = NNfuntion(normalized_array, list(range(190, 370)))

#wisper comparison 
anotherone =  [1.2469, 16.66 ,1.428 ,3.64 ,7.28 ,86.17391173, 37.98546466 ,6.133274559 ,53.20686303 ,3.473668347 ,4.212476735 ,92.6214029 ,6.530905501 ,6.220531832 ,73.13032831]
j=[330.020723, 324.9779508, 334.6890675, 322.1161648, 286.3767389, 248.89,262.72,282.81,267.21,354.99,356.19,280.85,354.64,355.15,284.50]

#wisper X comparison
#j = [287.4728191 ,289.8335315, 324.3892671, 363.6133094 ,279.7222718 , 287.3877785 ,367.0357938 ,322.7091633 ,273.126245 , 247.24, 277.04, 249.26, 246.32]
#anotherone= [10.97007248 ,9.303561687 ,5.361390383 ,2.447198192 ,17.1657704 ,12.4482893 ,1.296157743 ,4.992284311 ,19.26997116 ,251.1299197 ,21.44259995 ,163.2227418 ,225.182137]

plt.plot(cyclesSN, stressesSN, color = "black", label='CLD curves')
plt.plot(cyclesNN, stressesNN, color = "red", label='PINN')
plt.scatter(anotherone, j, color = "blue", label='Optidat values')
plt.ylabel("Max stress[MPA]", fontsize = 12)
plt.xlabel("Number of passes of Wisper Spectrum", fontsize = 12)
plt.xscale("log")
plt.legend()
plt.show()

cyclesSN, stressesSN = SNfunction(normalized_array, j, surface)
cyclesNN, stressesNN = NNfuntion(normalized_array, j)


# Sample array
data = [cyclesSN]

# Convert array to DataFrame
df = pd.DataFrame(data).transpose() 
# Define the Excel file path
excel_file = "stressesSN.xlsx"

# Write DataFrame to Excel
df.to_excel(excel_file, index=False, header=False)

print("Excel file created successfully.")

data = [cyclesNN]

# Convert array to DataFrame
df = pd.DataFrame(data).transpose() 

# Define the Excel file path
excel_file = "cyclesNN.xlsx"

# Write DataFrame to Excel
df.to_excel(excel_file, index=False, header=False)

print("Excel file created successfully.")


print("SN results")
print(cyclesSN)
a = []
b = []
c = []
for i in range(len(stressesSN)):
    a.append(np.abs(anotherone[i]-cyclesSN[i]))
    b.append((anotherone[i]-cyclesSN[i])**2)  
    c.append(np.abs(anotherone[i]-cyclesSN[i]))
a = np.average(a)
b = np.average(b)
c = np.max(c)
print("a")
print(a)
print(b)
print(c)

print("NN results")
print(cyclesNN)
a = []
b = []
c = []
for i in range(len(stressesNN)):
    a.append(np.abs(anotherone[i]-cyclesNN[i]))
    b.append((anotherone[i]-cyclesNN[i])**2)  
    c.append(np.abs(anotherone[i]-cyclesNN[i]))
a = np.average(a)
b = np.average(b)
c = np.max(c)
print("a")
print(a)
print(b)
print(c)

print("stressesSN")
print(cyclesSN)
print("stressesNN")
print(cyclesNN)
plt.xscale("log")
plt.yscale("log")
plt.scatter(anotherone, cyclesSN,  color = "black", label = "CLD")
plt.scatter(anotherone, cyclesNN, color = "blue", label = "PINN")

plt.ylabel("Number of passes CLD-PINN", fontsize = 12)
plt.xlabel("Number of passes  Optidat", fontsize = 12)
plt.legend()
plt.ylim(0.001, 10000)
plt.xlim(0.001, 10000)
x_values = np.linspace(-10, 300, 200)  # Creating an array of x values from 0 to 5
y_values = x_values  # Since x = y, y values will be the same as x values
plt.plot(x_values, y_values, color='red', linestyle='--')
plt.show()