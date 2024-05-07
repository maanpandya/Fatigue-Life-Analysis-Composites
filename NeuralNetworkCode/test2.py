import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
import time
import random as rd


path = 'NNModelArchive/rev3/fatstat420'
model, scaler = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')
data = dp.dfread(path + '/data.csv')
print(scaler['Ncycles'])
f.test_model(model, scaler, x_test, y_test)
#sn curve
if 'Cut angle ' in data.columns:
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
    targetR = -0.
    while True:
        indexes = list(x_test.index)
        if targetR is not None:
            indexes = list(data.loc[data['R-value1'] == targetR].index)
        i = rd.choice(indexes)
        datapoint = data.loc[i]
        R = datapoint['R-value1']
        f.sncurvereal(data, R)
        f.sncurvetest(model, 5, i, scaler, orig_data=data)
else:
    while True:
        indexes = list(x_test.index)
        i = rd.choice(indexes)
        datapoint = data.loc[i]
        f.sncurverealbasic(data)
        f.sncurvetest(model, 5, i, scaler, orig_data=data)



