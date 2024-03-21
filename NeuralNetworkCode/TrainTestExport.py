import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
from PINNLoss import PINNLoss

print('cuda available: ' + str(torch.cuda.is_available()))

# random seed
random_seed = False
if not random_seed:
    seed = 854616619
    torch.manual_seed(seed)
    np.random.seed(seed)

# input data
file = 'data3'
folder = 'DataProcessing/processed'
target_columns = ['Ncycles']            # max of 1 output
test_size = 0.3

# model parameters
n_hidden_layers = 0                           # int (set to zero to use len(layer_sizes)
layer_sizes = [128, 128, 256, 256, 128, 128]                              # int or list of int
act_fn = nn.Tanh()                    # fn or list of fn

# training parameters
n_epochs = 2000
loss_fn = PINNLoss            # fn
learning_rate = 0.0001
optimizer = torch.optim.Adam            # fn

# data loading
path = folder + '/' + file + '.csv'
data = dp.dfread(path)

# data changes
#data = dp.col_filter(data, ['Temp.'], 'exclude')

# data splits
traindata, testdata, scalers = dp.datasplitscale(data, test_size=test_size, exclude_columns=[])
x_train, y_train = dp.dfxysplit(traindata, target_columns)
x_test, y_test = dp.dfxysplit(testdata, target_columns)

# create model
if n_hidden_layers == 0:
    n_hidden_layers = len(layer_sizes)
n_inputs = len(x_train.columns)
n_outputs = len(y_train.columns)
model = f.create_model_2(n_inputs, layer_sizes, n_outputs, n_hidden_layers, act_fn)
model = model.double()
model.to('cuda')

# train
model = f.train_model(model, loss_fn, optimizer, n_epochs, learning_rate, x_train, y_train)

# test
f.test_model(model, scalers, x_test, y_test)

# export
name = input('enter a name to export model, enter <t> to use timetag: ')
if name != '':
    if name == 't':
        name = None
    f.export_model(model, 'NNModelArchive/rev2', scalers, name=name, data=data,
                   x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train)


