import sys
sys.path.append('NeuralNetworkCode')

import numpy as np
import rainflow 
import matplotlib.pyplot as plt
import function as f
import torch
import pandas as pd

path = 'NeuralNetworkCode/NNModelArchive/rev2/10x30pinloss'
model, scaler = f.import_model(path)

# Read the file
with open('NeuralNetworkCode\Wisper\WISPER', 'r') as file:
    file_contents = file.readlines()
array = np.array([float(x) for x in file_contents[1:]])

# Normalize the array to the maximum value
array = array-25
xddd = []
for i in range(len(array)):
    xddd.append(i)

plt.plot(xddd, array, color = "black")
plt.xlabel('Cycle Number', fontsize = 16)
plt.ylabel('Load Level', fontsize = 16)

# Title of the plot
plt.gcf().set_size_inches(10, 6)
plt.show()
normalized_array = array/ np.max(array)

# Title of the plot
plt.title('Whisper X Spectrum')
plt.show()
# Count cycles
counted_cycles = rainflow.count_cycles(normalized_array)
# Calculate the accumulated damage
stresses = []
cycles = []
data2 = pd.read_csv('NeuralNetworkCode/DataProcessing/processed/data2.csv')
x = pd.DataFrame(np.nan,index=[0],columns=data2.columns)
x['Fibre Volume Fraction'] =50.92 #Fibre Volume Fraction
x['Cut angle '] = 0 #Cut angle
x['taverage'] = 3.83 #Average thickness
x['waverage'] = 24.86 #Average width
x['area'] = 95.21 #Area
x['Lnominal'] = 145 #Nominal length of sample
#x['R-value1'] = -1 #R-value
x['Ffatigue'] = 36.08 #Fatigue force
x['f'] = 3.44 #Frequency
x['E'] = 37.46 #Young's modulus
x['Temp.'] = 28 #Temperature
#x["tens"] = 75
#x["comp"]= -45
x.drop(columns=['nr','Ncycles'],inplace=True)

#Normalize the data except for stress
for i in x.columns:
    if i != 'smax' and i != 'R-value1':
        x[i] = (x[i] - scaler[i]['mean']) / scaler[i]['std']

count_2 = 0
rng_list = []
R_list = []
xd  = []
for rng, mean, count, i_start, i_end in rainflow.extract_cycles(normalized_array):
    count_2 += 1 
    rng_list.append(rng)
    R = (mean- rng/2) / (mean + rng/2) 
    R_list.append(R)
    xd.append(count_2)

plt.plot(xd, R_list)
plt.show()

R_counted = []
Rng_counted = []
n_times_apeared = []
for i in range(172):
    n_times_apeared.append(0)
appered = 0 
j = 0

for i in range(len(rng_list)):
    appered = 0
    for j in range(len(R_counted)):
        if rng_list[i] == Rng_counted[j] and R_list[i] == R_counted[j] and appered == 0:
            n_times_apeared[j] = n_times_apeared[j] +1
            appered = 1
    if appered == 0:
        R_counted.append(R_list[i])
        n_times_apeared[j] = n_times_apeared[j] +1
        Rng_counted.append(rng_list[i])

plt.bar(Rng_counted, n_times_apeared)
plt.xlabel('Rng_counted')
plt.ylabel('n_times_appeared')
plt.title('Histogram of Rng_counted with n_times_appeared')
plt.show()

for j in [378.90 , 371.15, 379.23, 355.52, 381.00, 365.03, 374.66, 376.13, 349.57, 469.89, 466.83, 470.06, 342.82, 443.18, 444.34, 446.66, 357.72, 356.78]:
    Accumulated_stress = 0
    for i in range(len(R_counted)):
        N_Cycle =n_times_apeared[i] # fatigue life at reference stress 
        x['smax'] = Rng_counted[i]*j# stress
        x['R-value1'] = 1
        #R_counted[i]
        #Normalize the data for smax
        x['smax'] = (x['smax'] - scaler['smax']['mean']) / scaler['smax']['std']
        x['R-value1'] = (x['R-value1'] - scaler['R-value1']['mean']) / scaler['R-value1']["std"]
        #Predict the number of cycles
        xtest = torch.tensor(x.values)
        xtest=xtest.cuda()
        N_AI = model(xtest)
        N_AI = N_AI.cpu().detach().numpy()
        N_AI = N_AI*scaler["Ncycles"]["std"]+scaler["Ncycles"]["mean"]
        N_AI1 = 10**N_AI[0][0]
        #N_AI1 = (Stress/(367.8))**(-1/0.078)
        Accumulated_stress += N_Cycle/N_AI1

    cycles.append(1/Accumulated_stress)
    stresses.append(j)
anotherone = [46.63400685, 30.53759847, 60.13090802 , 86.38254409, 30.11657662, 36.04822509, 25.01408323, 24.02818907, 42.03037427, 2.819182283, 8.159798359, 3.860663396 , 65.88812532 ,11.30806298, 11.0245947, 10.94091318, 69.13812292, 83.13420841 ]
plt.scatter(np.log(anotherone), np.log(stresses), color = "black")
plt.xlabel("Logarithm scale number of cycles Optidat", fontsize = 12)
plt.ylabel("Logarithm scale number of cycles Wisper-PINN", fontsize = 12)
plt.xlim(0, 10)
plt.ylim(0, 10)
x_values = np.linspace(0, 10, 200)  # Creating an array of x values from 0 to 5
y_values = x_values  # Since x = y, y values will be the same as x values


plt.plot(x_values, y_values, color='red', linestyle='--')

plt.show()