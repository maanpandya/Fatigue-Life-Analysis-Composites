import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import DataProcessing.DPfunctions as dp
from PINNLoss import PINNLoss
import DataProcessing.DPfunctions as dp
import os
import pickle
import math

def create_model(n_inputs, layers=None, n_outputs=1):
    if layers is None:
        layers = [10, 10, 10, 10, 10]
    if len(layers) != 5:
        raise 'incompatable layer list'

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.dummy_param = nn.Parameter(torch.empty(0))
            self.layer1 = nn.Linear(n_inputs, layers[0])
            self.layer2 = nn.Linear(layers[0], layers[1])
            self.layer3 = nn.Linear(layers[1], layers[2])
            self.layer4 = nn.Linear(layers[2], layers[3])
            self.layer5 = nn.Linear(layers[3], layers[4])
            self.layer6 = nn.Linear(layers[4], n_outputs)

        def forward(self, x):
            device = self.dummy_param.device
            x = torch.sigmoid(self.layer1(x))
            x = torch.sigmoid(self.layer2(x))
            x = torch.relu(self.layer3(x))
            x = torch.relu(self.layer4(x))
            x = torch.relu(self.layer5(x))
            x = self.layer6(x)
            return x

    # Load the model
    model = NeuralNetwork().double()
    return model


def create_model_2(n_inputs, layer_sizes, n_outputs, n_hidden_layers, act_fn):
    if type(layer_sizes) != list:
        layer_sizes = n_hidden_layers * [layer_sizes]
    if type(act_fn) != list:
        act_fn = n_hidden_layers * [act_fn]

    # Define a list to hold the layers
    layers = []

    # Add input layer
    layers.append(nn.Linear(n_inputs, layer_sizes[0]))
    layers.append(act_fn[0])

    # Add hidden layers
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(act_fn[i + 1])

    # Add output layer
    layers.append(nn.Linear(layer_sizes[-1], n_outputs))

    # Combine all layers into a sequential model
    model = nn.Sequential(*layers)

    return model


def train_model(model, loss_fn, optimizer, n_epochs, learning_rate, x_train, y_train):
    X = torch.tensor(x_train.iloc[:, :len(x_train.columns)].values)
    X = X.cuda()
    print(X.device)
    X.requires_grad = True

    # Extract the output data from the last column
    y = torch.tensor(y_train.iloc[:, -1].values).view(-1, 1)
    y = y.cuda()
    print(y.device)


    losses = []
    t = time.time()
    n = 10
    model.train()

    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model(X)

        # Compute the loss
        if loss_fn == PINNLoss:
            loss = loss_fn(y_pred, y, X)
        else:
            loss = loss_fn(y_pred, y)
        losses.append(loss.item())

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # show progress
        progress = int((epoch / n_epochs) * 100)

        if progress - n >= 0:
            n = n + 10
            print('training progress: '+str(progress)+'%')
            print('time remaining: ' + str(((time.time()-t) / progress) * (100 - progress)) + 's')
    print('done in ' + str(time.time()-t))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('epoch = ' + str(n_epochs) + ', lr = ' + str(learning_rate))
    plt.show()
    return model


def test_model(model, scaler, x_test, y_test):
    model.eval()
    X_test = torch.tensor(x_test.iloc[:, :len(x_test.columns)].values)
    X_test = X_test.cuda()

    X_test.requires_grad = True

    Y_test = torch.tensor(y_test.iloc[:, -1].values).view(-1, 1)
    Y_test = Y_test.cuda()
    y_test_pred = model(X_test)
    pred_eval = pd.DataFrame(y_test_pred.cpu().detach().numpy()).set_index(y_test.index)
    pred_eval = pred_eval.rename(columns={pred_eval.columns[-1]: 'pred_scaled'})
    pred_eval = pred_eval.join(y_test)
    pred_eval = pred_eval.rename(columns={pred_eval.columns[-1]: 'real_scaled'})
    scaler = scaler[y_test.columns[0]]
    pred_eval['pred_log'] = pred_eval['pred_scaled'] * scaler['std'] + scaler['mean']
    pred_eval['real_log'] = pred_eval['real_scaled'] * scaler['std'] + scaler['mean']
    pred_eval['pred'] = np.power(10, pred_eval['pred_log'])
    pred_eval['real'] = np.power(10, pred_eval['real_log'])
    print(pred_eval)
    print('lMSE = ' + str(np.mean(np.power(pred_eval['pred_log'] - pred_eval['real_log'], 2))))
    print('lRMSE = ' + str(np.sqrt(np.mean(np.power(pred_eval['pred_log'] - pred_eval['real_log'], 2)))))
    print('lMAE = ' + str(np.mean(np.abs(pred_eval['pred_log'] - pred_eval['real_log']))))
    lMRE = np.abs((pred_eval['pred_log'] - pred_eval['real_log']) / (pred_eval['real_log']))
    lMRE = lMRE.replace([np.inf, -np.inf], 0)
    print('lMRE = ' + str(np.mean(lMRE)))
    #some values are inf, maybe outliers
    print('MRE = ' + str(np.mean(np.abs((pred_eval['pred'] - pred_eval['real']) / (pred_eval['real'])))))
    plt.scatter(pred_eval['real_log'], pred_eval['pred_log'])
    plt.plot([0, 10], [0, 10], color='red', linestyle='--')
    plt.xlabel('y_test')
    plt.ylabel('predicted')
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def sncurvetest(model, maxstressratio, dataindex, scalers, exportdata=False):
    data = dp.dfread("NeuralNetworkCode/DataProcessing/processed/data2.csv")
    data = data[dataindex:dataindex+1]
    data = data.drop(columns=['Ncycles'])
    smax = data['smax']
    data['smax']=0

    #Let x be a dataframe with the same columns as data but empty
    x = pd.DataFrame(columns=data.columns)
    #Keep increasing smax from 0 to the initial smax and appending the data to x
    iterations = maxstressratio
    for i in range(math.ceil(smax)*iterations):
        data['smax'] = int(i)
        #Append the data to the dataframe x as a row
        x = pd.concat([x, data])
        x['smax'] = x['smax'].astype(float)
    
    xorig = x.copy()

    print(x)

    #Scale x using the values in scalers
    for i in x.columns:
        x[i] = (x[i] - scalers[i]['mean']) / scalers[i]['std']

    print(x)
    #Predict the number of cycles
    model.eval()
    #print dtype of x
    print(x.dtypes)
    x = torch.tensor(x.values)
    x = x.cuda()
    x.requires_grad = True
    y = model(x)
    #Unscale the predicted number of cycles
    y = y.cpu().detach().numpy()
    y = y * scalers['Ncycles']['std'] + scalers['Ncycles']['mean']

    #Plot the results, the column smax from x on the y axis and the predicted number of cycles on the x axis
    plt.scatter(y, xorig['smax'])
    plt.xlabel('log of Ncycles')
    plt.ylabel('smax')
    plt.show()

def export_model(model, folder, scalers=None, name=None, x_train=None, y_train=None, x_test=None, y_test=None, data=None):
    if name == None:
        name = dp.timetag()
    path = folder + '/' + name
    os.makedirs(path)
    path = path + '/'
    model_scripted = torch.jit.script(model)
    model_scripted.save(path + 'model.pt')
    if type(x_train) == pd.DataFrame:
        pd.DataFrame.to_csv(x_train, path + 'x_train.csv')
    if type(y_train) == pd.DataFrame:
        pd.DataFrame.to_csv(y_train, path + 'y_train.csv')
    if type(x_test) == pd.DataFrame:
        pd.DataFrame.to_csv(x_test, path + 'x_test.csv')
    if type(y_test) == pd.DataFrame:
        pd.DataFrame.to_csv(y_test, path + 'y_test.csv')
    if type(data) == pd.DataFrame:
        pd.DataFrame.to_csv(data, path + 'data.csv')
    if not scalers == None:
        with open(path+'scalers.pkl', 'wb') as t:
            pickle.dump(scalers, t)
def import_model(path):
    path = path + '/'
    model = torch.jit.load(path + 'model.pt')
    with open(path + 'scalers.pkl', 'rb') as t:
        scaler = pickle.load(t)
    return model, scaler

def scale(data, scaler):
    for i in data.columns:
        data[i] = (data[i] - scaler[i]['mean']) / scaler[i]['std']
    return data

def inv_scale(data, scaler):
    for i in data.columns:
        data[i] = data[i] * scaler[i]['std'] + scaler[i]['mean']
    return data