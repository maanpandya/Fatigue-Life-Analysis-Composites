import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
sys.path.append('NeuralNetworkCode')

import numpy as np
import matplotlib.pyplot as plt
import function as f
import torch
import pandas as pd

def SNcurve1(x, y): #S-N Curve of the 1st block with R1 value
    return math.e**((x - 552.01)/-26.51)
def SNcurve2(x, y): #S-N Curve of the 2nd block with R2 value
    return math.e**((x - 552.01)/-26.51)
def NNmodel(x_imput):    
    path = 'NeuralNetworkCode/NNModelArchive/rev2/10x30pinloss'
    model, scaler = f.import_model(path)
    data2 = pd.read_csv('NeuralNetworkCode/DataProcessing/processed/data2.csv')
    x = pd.DataFrame(np.nan,index=[0],columns=data2.columns)
    x['Fibre Volume Fraction'] =50.92 #Fibre Volume Fraction
    x['Cut angle '] = 0 #Cut angle
    x['taverage'] = 3.83 #Average thickness
    x['waverage'] = 24.86 #Average width
    x['area'] = 95.21 #Area
    x['Lnominal'] = 145 #Nominal length of sample
    x['R-value1'] = -1 #R-value
    x['Ffatigue'] = 36.08 #Fatigue force
    x['f'] = 3.44 #Frequency
    x['E'] = 37.46 #Young's modulus
    x['Temp.'] = 28 #Temperature
    x['smax'] = x_imput
    #x["tens"] = 75
    #x["comp"]= -45

    x.drop(columns=['nr','Ncycles'],inplace=True)

    for i in x.columns:
        x[i] = (x[i] - scaler[i]['mean']) / scaler[i]['std']
    xtest = torch.tensor(x.values)
    xtest=xtest.cuda()
    N_AI = model(xtest)
    N_AI = N_AI.cpu().detach().numpy()
    N_AI = N_AI*scaler["Ncycles"]["std"]+scaler["Ncycles"]["mean"]
    N_AI1 = 10**N_AI[0][0]
    #N_AI1 = (Stress/(367.8))**(-1/0.078)
    return N_AI1
    
    





def Calculations(x, y)
letter_lst = []
for letter in code:
    letter_lst.append(letter)

if letter_lst[0] == 'b' : #normal block test
    if letter_lst[3] == '1' :
        N1 = 2500
    if letter_lst[3] == '2' :
        N1 = 25000
    if letter_lst[3] == '3' :
        N1 = 5e5
    #number of cycles till damage = 1
    N2 = []
    for i in range(len(Smax2)):
        for k in range(len(Smax1)):
            N2.append(SNcurve2(Smax2[i]) * ( 1 - (N1 / SNcurve1(Smax1[k]))))
    #deltaN2 = STDEV2 + (SNcurve2(Smax2) * N1 / SNcurve1(Smax1)) * (STDEV2/SNcurve2(Smax2) + STDEV1/SNcurve1(Smax1)))
            print(f'The material tested with {code} experiences {N2[i]}  cycles at a first load level of {Smax2[i]} MPa and a second load level of {Smax2[i]} MPa')
            

    """print(SNcurve2(Smax2))
    print(SNcurve1(Smax1))
    print(N1 / SNcurve1(Smax1))
    print(N1)"""


if letter_lst[0] == 'r' : #repeated block test
    if letter_lst[4] == '1' :
        N1 = 25
        if letter_lst[6] == '1' :
            N2 = 25
        if letter_lst[6] == '2' :
            N2 = 250
        if letter_lst[6] == '3' :
            N2 = 5000
    if letter_lst[4] == '2' :
        N1 = 250
    if letter_lst[4] == '3' :
        N1 = 5000
    if letter_lst[5] == '1' :
        N2 = 25
    if letter_lst[5] == '2' :
        N2 = 250
    if letter_lst[5] == '3' :
        N2 = 5000
    cycles1 =  np.zeros((len(Smax2), len(Smax2)))
    
    cycles2 =  np.zeros((len(Smax2), len(Smax2)))

    for i in range(len(Smax2)):
        for k in range(len(Smax1)):
            damage1 = N1 / SNcurve1(Smax1[k], 0) + N2 / SNcurve2(Smax2[i],0)
            damage2 = N1 / NNmodel(Smax1[k]) + N2 / NNmodel(Smax2[i])
            cycles1[i][k] = (1 / damage1)*(N1+N2)
            cycles2[i][k] = 1 / damage2*(N1+N2)

            print("This is the iteration number")
            print((i+1)*(k+1))
            print("The damage")
            print(damage1)
            print("and")
            print(damage2)
    # Convert Smax1 and Smax2 to numpy arrays for plotting
    Smax1_array = np.array(Smax1)
    print(Smax1_array)
    Smax2_array = np.array(Smax2)
    print(Smax2_array)

    x, y = np.meshgrid(Smax1_array, Smax2_array)
    cycles1_array = np.array(cycles1)
    cycles2_array = np.array(cycles2)
    print(cycles1_array)
    print(cycles2_array)