import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f

print('cuda available: ' + str(torch.cuda.is_available()))

# input data
file = 'data2'
folder = 'DataProcessing/processed'
target_columns = ['Ncycles']
test_size = 0.2

# model parameters
hidden_layers = 5                              # int
n_neurons = 20                            # int or list
act_fn = torch.sigmoid                  # fn or list of fn

# training parameters
n_epochs = 10000
loss_fn = nn.MSELoss                    # fn
optimizer = torch.optim.Adam            # fn

# data loading
path = folder + '/' + file + '.csv'
data = dp.dfread(path)
traindata, testdata, scalers = dp.datasplitscale(data, test_size=test_size, exclude_columns=[])
x_train, y_train = dp.dfxysplit(traindata, target_columns)
x_test, y_test = dp.dfxysplit(testdata, target_columns)
print(data)
print(traindata)
print(x_train)
print(y_train)

# create model
n_inputs = len(x_train.columns)
n_outputs = len(y_train.columns)

f.create_model_2(n_inputs, hidden_layers, n_neurons, n_outputs, act_fn)