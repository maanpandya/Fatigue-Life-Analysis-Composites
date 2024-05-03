import sys
sys.path.append('NeuralNetworkCode')

import numpy as np
import rainflow 
import matplotlib.pyplot as plt
import function as f
import torch
import pandas as pd

def NNfuntion(normalized_array, wawas):


    path = 'NeuralNetworkCode/NNModelArchive/rev2/10x30pinloss'
    model, scaler = f.import_model(path)
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
        if i != 'smax' and i != 'R-value1' and i != 'Ffatigue':
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

    #for j in [378.90 , 371.15, 379.23, 355.52, 381.00, 365.03, 374.66, 376.13, 349.57, 469.89, 466.83, 470.06, 342.82, 443.18, 444.34, 446.66, 357.72, 356.78]:
    for j in wawas:
        Accumulated_stress = 0
        for i in range(len(R_counted)):
            N_Cycle =n_times_apeared[i] # fatigue life at reference stress 
            x['smax'] = Rng_counted[i]*j# stress
            x['Ffatigue'] = Rng_counted[i]*j*95.21/1000
            x['R-value1'] = R_counted[i]
            #Normalize the data for smax
            x['Ffatigue'] = (x['Ffatigue'] - scaler['Ffatigue']['mean']) / scaler['Ffatigue']['std']
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
    return cycles, stresses
