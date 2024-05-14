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
from DPfunctions import rmath

import CLD_interpolator
from CLD_interpolator import CLD_interpolator_log

def SNcurve1(x, R_value, suface): #S-N Curve of the 1st block with R1 value
    return CLD_interpolator_log(suface, x*(1-R_value)/2, R_value )
def NNmodel(x_imput, R_value):    
    path = 'NeuralNetworkCode/NNModelArchive/finalmodels/correctsmax3'
    model, scaler = f.import_model(path)
    data2 = pd.read_csv('NeuralNetworkCode/DataProcessing/processed/data12.csv')
    x = pd.DataFrame(np.nan,index=[0],columns=data2.columns)
    #x['Fibre Volume Fraction'] =50.92 #Fibre Volume Fraction
    #x['Cut angle '] = 0 #Cut angle
    x['taverage'] = 6.6 #Average thickness
    x['waverage'] = 25#Average width
    x['area'] = 162.31 #Area
    x['Lnominal'] = 150 #Nominal length of sample
    #x['R-value1'] = -1 #R-value
    #x['Fmax'] = 36.08 #Fatigue force
    x['smax'] = x_imput
    x['smean'] = 0
    x['smin'] = 0
    #x['f'] = 3.44 #Frequency
    #x['E'] = 37.46 #Young's modulus
    #x['Temp.'] = 28 #Temperature
    #x["tens"] = 75
    #x["comp"]= -45
    x.drop(columns=['nr','Ncycles'],inplace=True)

    x['smean'] = rmath({'smax':x['smax'], 'R':R_value}, 'smean')
    x['smin'] = rmath({'smax':x['smax'], 'R':R_value}, 'smin')
    for i in x.columns:
        x[i] = (x[i] - scaler[i]['mean']) / scaler[i]['std']
    print(x)
    xtest = torch.tensor(x.values)
    xtest=xtest.cuda()
    N_AI = model(xtest)
    N_AI = N_AI.cpu().detach().numpy()
    N_AI = N_AI*scaler["Ncycles"]["std"]+scaler["Ncycles"]["mean"]
    N_AI1 = 10**N_AI[0][0]
    #N_AI1 = (Stress/(367.8))**(-1/0.078)
    return N_AI1
    
    





def Calculations(Smax1, Smax2, code, surface, R):
    letter_lst = []
    for letter in code:
        letter_lst.append(letter)

    if letter_lst[0] == 'b' : #normal block test
        
        cycles1 =  np.zeros((len(Smax2), len(Smax2)))
        cycles2 =  np.zeros((len(Smax2), len(Smax2)))
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
                if ( 1 - (N1 /SNcurve1(Smax1[k],  R, surface))) > 0:
                    cycles1[i][k] = (SNcurve1(Smax2[i], R, surface) * ( 1 - (N1 /SNcurve1(Smax1[k],  R, surface))))
                else:
                    cycles1[i][k] = 1
                cycles2[i][k] = (NNmodel(Smax2[i], R) * ( 1 - (N1 / NNmodel(Smax1[k],  R))))

        cycles1_array = np.array(cycles1)
        cycles2_array = np.array(cycles2)
        Smax1_array = np.array(Smax1)
        Smax2_array = np.array(Smax2)

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
                damage1 = N1 / SNcurve1(Smax1[k], R, surface) + N2 / SNcurve1(Smax2[i], R, surface)
                damage2 = N1 / NNmodel(Smax1[k], R) + N2 / NNmodel(Smax2[k], R)
                cycles1[i][k] = (1 / damage1)*(N1+N2)
                cycles2[i][k] = 1 / damage2*(N1+N2)

        # Convert Smax1 and Smax2 to numpy arrays for plotting
        Smax1_array = np.array(Smax1)
        Smax2_array = np.array(Smax2)
        cycles1_array = np.array(cycles1)
        cycles2_array = np.array(cycles2)

        
    x, y = np.meshgrid(Smax1_array, Smax2_array)
    return cycles1_array, cycles2_array, x, y