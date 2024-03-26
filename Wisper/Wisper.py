import sys
sys.path.append('NeuralNetworkCode/DataProcessing')

import numpy as np
import rainflow 
import matplotlib.pyplot as plt
import function as f
import torch
import pandas as pd
import DataProcessing.DPfunctions as dp 

path = 'NeuralNetworkCode/NNModelArchive/rev2/10x30pinloss'
model, scaler = f.import_model(path)

# Read the file
with open('Wisper/WISPER', 'r') as file:
    file_contents = file.readlines()
array = np.array([float(x) for x in file_contents[1:]])

# Normalize the array to the maximum value
normalized_array = array / np.max(array)

# Count cycles
counted_cycles = rainflow.count_cycles(normalized_array)

# Calculate the accumulated damage
Max_Force = 330 # Mpa

print(counted_cycles)
Accumulated_stress = 0
for i in range (len(counted_cycles)):
    N_Cycle =counted_cycles[i][1] # fatigue life at reference stress 
    Stress = counted_cycles[i][0] * Max_Force# stress
    B = 1.69111*10**22
    #N_AI = -np.log(Stress/178.08)*10**(-7)

    #Number of cycles predicted by AI model
    #N_AI = (Stress/(367.8))**(-1/0.078)
    data2 = pd.read_csv('NeuralNetworkCode/DataProcessing/processed/testdata2.csv')
    x = pd.DataFrame(columns=data2.columns)
    x['smax'] = Stress#Max Stress
    x['Fibre Volume Fraction'] =53.26 #Fibre Volume Fraction
    x['Cut angle'] = 0#Cut angle
    x['taverage'] = 6.45#Average thickness
    x['waverage'] = 25.25#Average width
    x['area'] = 162.86#Area
    x['Lnominal'] = 150#Nominal length of sample
    x['R-value1'] = -1#R-value
    x['Ffatigue'] = 53.75#Fatigue force
    x['f'] = 363#Frequency
    x['E'] = 30#Young's modulus
    x['Temp'] = 28 #Temperature

    #Normalize the data
    for i in x.columns:
        x[i] = (x[i] - scalers[i]['mean']) / scalers[i]['std']
    
    #Predict the number of cycles
    x = torch.tensor(x.values)
    model.eval()
    N_AI = model(x)
    N_AI = N_AI.cpu().detach().numpy()

    Accumulated_stress += N_Cycle/N_AI

stresses = []
cycles = []
for j in range(340):
    Max_Force = j
    Accumulated_stress = 0
    for i in range (len(counted_cycles)):
        N_Cycle =counted_cycles[i][1] # fatigue life at reference stress 
        Stress = counted_cycles[i][0] * Max_Force# stress
        N_AI = (Stress/(367.8))**(-1/0.078)
        Accumulated_stress += N_Cycle/N_AI
        
    cycles.append(1/Accumulated_stress)
    stresses.append(j)
plt.plot(np.log(cycles), stresses)  # Changed the order of the axes
plt.xlabel('N_AI')  # Updated the x-axis label
plt.ylabel('Stress')  # Updated the y-axis label
plt.title('N_AI vs Stress')
plt.show()



# Calculate N_AI vs stress
stress_values = [counted_cycles[i][0] * Max_Force for i in range(len(counted_cycles))]
N_AI_values = [(Stress / 624.49) ** (-1 / 0.088) for Stress in stress_values]

# Plot N_AI vs stress
plt.plot(N_AI_values, stress_values)  # Changed the order of the axes
plt.xlabel('N_AI')  # Updated the x-axis label
plt.ylabel('Stress')  # Updated the y-axis label
plt.title('N_AI vs Stress')
plt.show()

# Sum up the damage - according to Miner's rule, failure occurs when this sum exceeds 1
