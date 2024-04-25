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
with open('NeuralNetworkCode\Wisper_X\WISPERX', 'r') as file:
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

# Count cycles
counted_cycles = rainflow.count_cycles(normalized_array)
# Calculate the accumulated damage
stresses = []
cycles = []
data2 = pd.read_csv('NeuralNetworkCode/DataProcessing/processed/data2.csv')
x = pd.DataFrame(np.nan,index=[0],columns=data2.columns)
x['Fibre Volume Fraction'] =54.31 #Fibre Volume Fraction
x['Cut angle '] = 0 #Cut angle
x['taverage'] = 3.55 #Average thickness
x['waverage'] = 24.88 #Average width
x['area'] = 88.06 #Area
x['Lnominal'] = 150 #Nominal length of sample
#x['R-value1'] = -1 #R-value
x['Ffatigue'] = 33.55 #Fatigue force
x['f'] = 3.63 #Frequency
x['E'] = 42 #Young's modulus
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
for i in range(118):
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

for j in range(0, 500):
    Accumulated_stress = 0
    for i in range(len(R_counted)):
        N_Cycle =n_times_apeared[i] # fatigue life at reference stress 
        x['smax'] = Rng_counted[i]*j# stress
        x['R-value1'] = R_counted[i]
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
    print(j)
    print(1/Accumulated_stress)

    cycles.append(1/Accumulated_stress)
    stresses.append(j)
plt.plot(np.log(cycles), stresses)
plt.show()
# Changed the order of the axes
# Sum up the damage - according to Miner's rule, failure occurs when this sum exceeds 1