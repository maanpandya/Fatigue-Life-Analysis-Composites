import sys
sys.path.append('NeuralNetworkCode')

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

stresses = []
cycles = []
data2 = pd.read_csv('NeuralNetworkCode/DataProcessing/processed/testdata2.csv')
x = pd.DataFrame(np.nan,index=[0],columns=data2.columns)
x['Fibre Volume Fraction'] =53.26 #Fibre Volume Fraction
x['Cut angle '] = 0 #Cut angle
x['taverage'] = 6.45 #Average thickness
x['waverage'] = 25.25 #Average width
x['area'] = 162.86 #Area
x['Lnominal'] = 150 #Nominal length of sample
x['R-value1'] = -1 #R-value
x['Ffatigue'] = 53.75 #Fatigue force
x['f'] = 3.63 #Frequency
x['E'] = 30 #Young's modulus
x['Temp.'] = 28 #Temperature
x.drop(columns=['nr','Ncycles'],inplace=True)

#Normalize the data except for stress
for i in x.columns:
    if i != 'smax':
        x[i] = (x[i] - scaler[i]['mean']) / scaler[i]['std']

for j in range(340):
    Accumulated_stress = 0
    for i in range (len(counted_cycles)):
        N_Cycle =counted_cycles[i][1] # fatigue life at reference stress 
        Stress = counted_cycles[i][0] * (j)# stress
        x['smax'] = Stress #Max Stress

        #Normalize the data for smax
        x['smax'] = (x['smax'] - scaler['smax']['mean']) / scaler['smax']['std']

        #Predict the number of cycles
        xtest = torch.tensor(x.values)

        xtest=xtest.cuda()
        N_AI = model(xtest)
        N_AI = N_AI.cpu().detach().numpy()
        N_AI1 = 10**N_AI[0][0]
        Accumulated_stress += N_Cycle/N_AI1
    print(j)
    print(1/Accumulated_stress)

    cycles.append(1/Accumulated_stress)
    stresses.append(j)
plt.plot(cycles, stresses)
plt.show()
  # Changed the order of the axes
# Sum up the damage - according to Miner's rule, failure occurs when this sum exceeds 1
