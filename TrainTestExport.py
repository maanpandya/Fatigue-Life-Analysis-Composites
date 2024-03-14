import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
from PINNLoss import PINNLoss

print('cuda available: ' + str(torch.cuda.is_available()))

# input data
file = 'data2'
folder = 'DataProcessing/processed'
target_columns = ['Ncycles']            # max of 1 output
test_size = 0.2

# model parameters
n_hidden_layers = 5                           # int
layer_sizes = 20                              # int or list of int
act_fn = nn.Tanh()                    # fn or list of fn

# training parameters
n_epochs = 1000
loss_fn = PINNLoss                    # fn
learning_rate = 0.01
optimizer = torch.optim.Adam            # fn

# data loading
path = folder + '/' + file + '.csv'
data = dp.dfread(path)
traindata, testdata, scalers = dp.datasplitscale(data, test_size=test_size, exclude_columns=[])
x_train, y_train = dp.dfxysplit(traindata, target_columns)
x_test, y_test = dp.dfxysplit(testdata, target_columns)

print(scalers[target_columns[0]])
# create model
n_inputs = len(x_train.columns)
n_outputs = len(y_train.columns)

model = f.create_model_2(n_inputs, layer_sizes, n_outputs, n_hidden_layers, act_fn)
model = model.double()
model.to('cuda')
model = f.train_model(model, loss_fn, optimizer, n_epochs, learning_rate, x_train, y_train)
f.test_model(model, loss_fn, scalers, x_test, y_test)