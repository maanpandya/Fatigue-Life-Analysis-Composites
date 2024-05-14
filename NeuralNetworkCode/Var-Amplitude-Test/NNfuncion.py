import sys
sys.path.append('NeuralNetworkCode')

import numpy as np
import rainflow 
import matplotlib.pyplot as plt
import function as f
import torch
import pandas as pd
from DPfunctions import rmath

def NNfuntion(normalized_array, wawas):


    path = 'NeuralNetworkCode/NNModelArchive/finalmodels/correctsmax3'
    model, scaler = f.import_model(path)
    stresses = []
    cycles = []
    data2 = pd.read_csv('NeuralNetworkCode/DataProcessing/processed/data12.csv')
    x = pd.DataFrame(np.nan,index=[0],columns=data2.columns)
    #x['Fibre Volume Fraction'] =50.92 #Fibre Volume Fraction
    #x['Cut angle '] = 0 #Cut angle
    x['taverage'] = 6.45 #Average thickness
    x['waverage'] = 25#Average width
    x['area'] = 160.00 #Area
    x['Lnominal'] = 150 #Nominal length of sample
    #x['R-value1'] = -1 #R-value
    #x['Fmax'] = 36.08 #Fatigue force
    x['smax'] = 0
    x['smean'] = 0
    x['smin'] = 0
    #x['f'] = 3.44 #Frequency
    #x['E'] = 37.46 #Young's modulus
    #x['Temp.'] = 28 #Temperature
    #x["tens"] = 75
    #x["comp"]= -45
    x.drop(columns=['nr','Ncycles'],inplace=True)

    #Normalize the data except for stress
    for i in x.columns:
        if i != 'smax' and i != 'R-value1' and i != 'Fmax' and i != 'smean':
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
    print(sum(n_times_apeared))
    a = []
    b = []
    c = []
    for i in range(len(R_counted)):
        a.append(Rng_counted[i])
        b.append(rmath({'smax':Rng_counted[i], 'R':R_counted[i]}, 'smean'))
        c.append(rmath({'smax':Rng_counted[i], 'R':R_counted[i]}, 'smin'))

    for j in wawas:
        Accumulated_stress = 0
        for i in range(len(R_counted)):
            N_Cycle =n_times_apeared[i] # fatigue life at reference stress 
            x['smax'] = Rng_counted[i]*j 
            x['smean'] =b[i]*j 
            x['smin'] = c[i]*j
            # stress
            #x['Fmax'] = Rng_counted[i]*j* 167.31/1000
            #x['R-value1'] = R_counted[i]
            #Normalize the data for smax
            #x['Fmax'] = (x['Fmax'] - scaler['Fmax']['mean']) / scaler['Fmax']['std']
            x['smax'] = (x['smax'] - scaler['smax']['mean']) / scaler['smax']['std']
            x['smean'] = (x['smean'] - scaler['smean']['mean']) / scaler['smean']['std']
            x['smin'] = (x['smin'] - scaler['smin']['mean']) / scaler['smin']['std']

            #Predict the number of cycles

            #print("--------------")
            
            xtest = torch.tensor(x.values)
            xtest=xtest.cuda()
            N_AI = model(xtest)
            N_AI = N_AI.cpu().detach().numpy()
            #print(N_AI)
            N_AI = N_AI*scaler["Ncycles"]["std"]+scaler["Ncycles"]["mean"]
            #print(N_AI)
            N_AI1 = 10**N_AI[0][0]
            #print(N_AI1)
            #print(N_Cycle)
            Accumulated_stress += N_Cycle/N_AI1
            #print("--------------")
        print(j)
        print(x)
        print(1/Accumulated_stress)
        cycles.append(1/Accumulated_stress)
        stresses.append(j)
    return cycles, stresses
