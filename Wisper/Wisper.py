import sys
sys.path.append('NeuralNetworkCode')

import numpy as np
import rainflow 
import matplotlib.pyplot as plt
import function as f
import torch
import pandas as pd
import DataProcessing.DPfunctions as dp

path = 'NeuralNetworkCode/NNModelArchive/rev2/0449mse4mrenoisetrained'
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

stresses = []
cycles = []
<<<<<<< HEAD
Stress_test = []
N_test = []
=======
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

>>>>>>> cf0efb2b49984b46a93ea3bcc700ce3dea624a13
for j in range(340):
    Accumulated_stress = 0
    for i in range (len(counted_cycles)):
        N_Cycle =counted_cycles[i][1] # fatigue life at reference stress 
<<<<<<< HEAD
        Stress = counted_cycles[i][0] * (340-j)# stress
        data2 = pd.read_csv('NeuralNetworkCode/DataProcessing/processed/traindata2.csv')
        x = pd.DataFrame(np.nan,index=[0],columns=data2.columns)
        x['smax'] = Stress #Max Stress
        x['Fibre Volume Fraction'] =53.26 #Fibre Volume Fraction
        x['Cut angle '] = 0 #Cut angle
        x['taverage'] = 6.45 #Average thickness
        x['waverage'] = 25.25 #Average width
        x['area'] = 162.86 #Area
        x['Lnominal'] = 150 #Nominal length of sample
        x['R-value1'] = -1 #R-value
        x['Ffatigue'] = 53.75 #Fatigue force
        x['f'] = 363 #Frequency
        x['E'] = 30 #Young's modulus
        x['Temp.'] = 28 #Temperature
        x.drop(columns=['nr','Ncycles'],inplace=True)

        #Normalize the data
        for i in x.columns:
            x[i] = (x[i] - scaler[i]['mean']) / scaler[i]['std']

        
=======
        Stress = counted_cycles[i][0] * (j)# stress
        print(Stress)
        B = 1.69111*10**22
        print(x)
        x['smax'] = Stress #Max Stress

        #Normalize the data for smax
        x['smax'] = (x['smax'] - scaler['smax']['mean']) / scaler['smax']['std']

>>>>>>> cf0efb2b49984b46a93ea3bcc700ce3dea624a13
        #Predict the number of cycles
        xtest = torch.tensor(x.values)

        xtest=xtest.cuda()
<<<<<<< HEAD
        print(x)
        model.eval()
        print(model(xtest))
        N_AI = model(xtest) 
        N_AI = N_AI.cpu().detach().item()
=======
        N_AI = model(xtest)
        N_AI = N_AI.cpu().detach().numpy()
        print(N_AI)
>>>>>>> cf0efb2b49984b46a93ea3bcc700ce3dea624a13
        N_AI = np.power(10, N_AI)
    
        N_test.append(N_AI)
        Stress_test.append(Stress)
        
        Accumulated_stress += N_Cycle/N_AI
<<<<<<< HEAD
    fig = plt.figure()
    print(N_test)
    print(Stress_test)
    plt.plot(N_test, Stress_test, color='tab:blue')  # Use plt.plot() instead of fig.plot()
    plt.show()
    print(1/Accumulated_stress)
=======
    
>>>>>>> cf0efb2b49984b46a93ea3bcc700ce3dea624a13
    cycles.append(1/Accumulated_stress)
    print(j)
    stresses.append(j)
plt.plot(cycles, stresses)  # Changed the order of the axes
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
