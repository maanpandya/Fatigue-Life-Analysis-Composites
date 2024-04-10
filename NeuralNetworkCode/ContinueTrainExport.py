import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
from customloss import PINNLoss

name = '128-256x6mre3v2'
folder = 'NNModelArchive/rev2'
path = folder + '/' + name

# training parameters
n_epochs = 1000
loss_fn = nn.MSELoss()            # fn
learning_rate = 0.0001
optimizer = torch.optim.Adam            # fn

model, scalers = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')
x_train = dp.dfread(path + '/x_train.csv')
y_train = dp.dfread(path + '/y_train.csv')
data = dp.dfread(path + '/data.csv')

f.test_model(model, scalers, x_test, y_test)

model = f.train_model(model, loss_fn, optimizer, n_epochs, learning_rate, x_train, y_train)

f.test_model(model, scalers, x_test, y_test)

namee = name + input('enter a name to export model, enter <t> to use timetag: ' + name)
if namee != name:
    if namee == name + 't':
        namee = name + dp.timetag()
    f.export_model(model, 'NNModelArchive/rev2', scalers, name=namee, data=data,
                   x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train)