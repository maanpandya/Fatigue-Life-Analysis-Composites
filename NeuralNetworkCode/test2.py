import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
import time
import random as rd

complete = True
path = 'NeuralNetworkCode/NNModelArchive/rev4/PINNnewlr0.0001epoch3000'
model, scaler = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')
data = dp.dfread(path + '/data.csv')
f.test_model(model, scaler, x_test, y_test)

Rlist = [-2.5, -1, -0.4, 0.1, 0.5, 2, 10]
print(Rlist)
while True:
    # generate sn curves for random geometry from dataset
    i = rd.choice(data.index)
    datapoint = data.loc[i]
    print(datapoint)
    datapoint = datapoint.to_frame().T
    for i in Rlist:
        f.complete_sncurve2(datapoint, data, i, model, scaler, minstress=0, maxstress=800, exp=True)
    plt.legend()
    plt.xlim(0,7)
    plt.show()


'''#sn curve
if complete:
    notR = False
    if 'R-value1' not in data.columns:
        data['R-value1'] = dp.rmath({'smax':data['smax'], 'smean':data['smean']}, 'R')
        notR = True
    targetR = 0.1
    while True:
        indexes = list(x_test.index)
        if 'R-value1' in data.columns:
            if targetR is not None:
                indexes = list(data.loc[data['R-value1'] == targetR].index)
            else:
                indexes = list(data.loc[data['R-value1'] != 0].index)
        i = rd.choice(indexes)
        if notR:
            data = dp.col_filter(data, ['R-value1'], 'exclude')
        datapoint = data.loc[i]
        datapoint = datapoint.to_frame().T
        print(datapoint)
        f.complete_sn_curve(model, scaler, data, datapoint)
elif 'Cut angle ' in data.columns:
    while True:
        indexes = list(x_test.index)
        ca = 90.0
        while ca != 0:
            i = rd.choice(indexes)
            datapoint = data.loc[i]
            ca = datapoint['Cut angle ']
        R = datapoint['R-value1']
        print(f"i = {i}, R = {R}, cut angle = {ca}")
        f.sncurvereal(data, R)
        f.sncurvetest(model, 5, i, scaler, orig_data=data)
elif 'R-value1' in data.columns:
    targetR = None
    while True:
        indexes = list(x_test.index)
        if targetR is not None:
            indexes = list(data.loc[data['R-value1'] == targetR].index)
        else:
            indexes = list(data.loc[data['R-value1'] != 0].index)
        i = rd.choice(indexes)
        datapoint = data.loc[i]
        R = datapoint['R-value1']
        datapoint = datapoint.to_frame().T
        f.sncurvereal(data, R)
        f.sncurvetest(model, 800, datapoint, scaler)
else:
    while True:
        indexes = list(x_test.index)
        i = rd.choice(indexes)
        datapoint = data.loc[i]
        f.sncurverealbasic(data)
        f.sncurvetest(model, 5, i, scaler, orig_data=data)'''



