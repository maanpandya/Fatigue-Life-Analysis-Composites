import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
import time
import random as rd


path = 'NNModelArchive/rev3/data6_0468'
model, scaler = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')
data = dp.dfread(path + '/data.csv')
#f.test_model(model, scaler, x_test, y_test)
#sn curve
indexes = list(x_test.index)
ca = 90.0
while ca != 0:
    i = rd.choice(indexes)
    datapoint = data.loc[i]
    ca = datapoint['Cut angle ']
print(f"R = {datapoint['R-value1']}, cut angle = {ca}")
f.sncurvetest(model, 5, i, scaler, orig_data=data)
# maybe this function gives an error when smax < 0 which goes together with r=10


