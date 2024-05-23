import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
import time
import random as rd


# main
path = 'NNModelArchive/finalmodels/correctsmax3'
name = path.split('/')[-1]
model, scaler = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')
data = dp.dfread(path + '/data.csv')
data = dp.dfread('NNModelArchive/finalmodels/testaltmse2' + '/data.csv')
f.test_model(model, scaler, x_test, y_test)

exp = True
compare = True
plot_abs = True
show_grad = False
# compare to
path2 = 'NNModelArchive/rev4/noSgoe2'
name2 = path2.split('/')[-1]
model2, scaler2 = f.import_model(path2)
x_test2 = dp.dfread(path2 + '/x_test.csv')
y_test2 = dp.dfread(path2 + '/y_test.csv')
data2 = dp.dfread(path2 + '/data.csv')
print()
Rlist = [-2.5, -1, -0.4, 0.1, 0.5, 2, 10]
if show_grad:
    Rlist = [0.5]
if compare:
    Rlist = [-1, 0.1]
while True:
    # generate sn curves for random geometry from dataset
    i = rd.choice(data.index)
    datapoint = data.loc[i]
    print(datapoint)
    datapoint = datapoint.to_frame().T
    if compare:
        if all(data.columns==data2.columns):
            datapoint2 = datapoint
        else:
            i = rd.choice(data2.index)
            datapoint2 = data2.loc[i]
            print(datapoint2)
            datapoint2 = datapoint2.to_frame().T
    for i in Rlist:
        color = f.randomcolor()
        f.complete_sncurve2(datapoint, data, i, model, scaler, minstress=0, maxstress=600, exp=exp, name=name, color=color, plot_abs=plot_abs, show_grad=show_grad)
        if compare:
            f.complete_sncurve2(datapoint2, data2, i, model2, scaler2, minstress=0, maxstress=600, exp=False, name=name2, color=color*0.6, plot_abs=plot_abs)
    plt.legend()
    plt.xlim(-3,10)
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



