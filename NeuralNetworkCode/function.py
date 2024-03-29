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
    #print(X.device)
    X.requires_grad = True

    # Extract the output data from the last column
    y = torch.tensor(y_train.iloc[:, -1].values).view(-1, 1)
    y = y.cuda()
    #print(y.device)

    print('Training starting...')
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
    y_test_pred = model(X_test)

    # create dataframe of data
    pred_eval = pd.DataFrame(y_test_pred.cpu().detach().numpy()).set_index(y_test.index)
    pred_eval = pred_eval.rename(columns={pred_eval.columns[-1]: 'pred_scaled'})
    pred_eval = pred_eval.join(y_test)
    pred_eval = pred_eval.rename(columns={pred_eval.columns[-1]: 'real_scaled'})
    scaler = scaler[y_test.columns[0]]
    pred_eval['pred_log'] = pred_eval['pred_scaled'] * scaler['std'] + scaler['mean']
    pred_eval['real_log'] = pred_eval['real_scaled'] * scaler['std'] + scaler['mean']
    pred_eval['pred'] = np.power(10, pred_eval['pred_log'])
    pred_eval['real'] = np.power(10, pred_eval['real_log'])

    # print various measures of accuracy
    print('Measures of error:')
    print('lMSE = ' + str(np.mean(np.power(pred_eval['pred_log'] - pred_eval['real_log'], 2))))
    print('lRMSE = ' + str(np.sqrt(np.mean(np.power(pred_eval['pred_log'] - pred_eval['real_log'], 2)))))
    print('lMAE = ' + str(np.mean(np.abs(pred_eval['pred_log'] - pred_eval['real_log']))))
    lMRE = np.abs((pred_eval['pred_log'] - pred_eval['real_log']) / (pred_eval['real_log']))
    a = 0
    for i in lMRE.index:
        if lMRE[i] == np.inf or lMRE[i] == -np.inf:
            a+=1
    if a/len(lMRE.index) > 0.1:
        print('a lot of inf in lMRE')
    lMRE = lMRE.replace([np.inf, -np.inf], np.nan)
    print('lMRE = ' + str(np.nanmean(lMRE)))
    print('MRE = ' + str(np.mean(np.abs((pred_eval['pred'] - pred_eval['real']) / (pred_eval['real'])))))

    # outlier detection
    lAE = np.abs(pred_eval['pred_log'] - pred_eval['real_log'])
    lAE = lAE.sort_values(ascending=False)
    lE = pred_eval['pred_log'] - pred_eval['real_log']
    print('top 5 lAE:')
    print(lE.loc[lAE.index[0:5:]])

    # plot errors
    plt.scatter(pred_eval['real_log'], pred_eval['pred_log'])
    plt.plot([-100, 100], [-100, 100], color='red', linestyle='--')
    plt.plot([-100, 100], [-101, 99], color='darkred', linestyle='--')
    plt.plot([-100, 100], [-99, 101], color='darkred', linestyle='--')
    plt.xlabel('y_test')
    plt.ylabel('predicted')
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def sncurvetest(model, maxstressratio, dataindex, scalers, testdatafile='data2.csv', exportdata=False):
    path = 'DataProcessing/processed/' + testdatafile
    data = dp.dfread(path)
    data = data[dataindex:dataindex+1]
    data = data.drop(columns=['Ncycles'])
    smax = data['smax']
    data['smax']=0

    #Let x be a dataframe with the same columns as data but empty
    x = pd.DataFrame(columns=data.columns)
    #Keep increasing smax from 0 to the initial smax and appending the data to x
    iterations = math.ceil(smax)*maxstressratio
    for i in range(iterations):
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
    #Set domain and range of the plot
    #Domain should be more than 0 and less than the maximum value of the predicted number of cycles
    #Range should be more than 0 and less than the maximum value of smax
    plt.xlim(0, np.max(y))
    plt.ylim(0, iterations)
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


def train_validate_model(model, loss_fn, optimizer, n_epochs, learning_rate, x_train, y_train, x_test, y_test, best=True):
    # slower, but shows test loss to find over fitting and picks best model of all epochs
    from copy import deepcopy
    # extract training data
    x_train = torch.tensor(x_train.iloc[:, :len(x_train.columns)].values)
    x_train = x_train.cuda()
    x_train.requires_grad = True
    y_train = torch.tensor(y_train.iloc[:, -1].values).view(-1, 1)
    y_train = y_train.cuda()

    # extract test / validation data
    x_test = torch.tensor(x_test.iloc[:, :len(x_test.columns)].values)
    x_test = x_test.cuda()
    x_test.requires_grad = True
    y_test = torch.tensor(y_test.iloc[:, -1].values).view(-1, 1)
    y_test = y_test.cuda()

    print('Training starting...')
    losses = []
    testlosses = []
    t = time.time()
    n = 10
    model.train()

    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(n_epochs):
        # Forward pass
        y_pred_train = model(x_train)
        y_pred_test = model(x_test)
        # Compute the loss
        if loss_fn == PINNLoss:
            loss = loss_fn(y_pred_train, y_train, x_train)
            testloss = loss_fn(y_pred_test, y_test, x_test)
        else:
            loss = loss_fn(y_pred_train, y_train)
            testloss = loss_fn(y_pred_test, y_test)
        losses.append(loss.item())
        testloss = testloss.item()
        testlosses.append(testloss)
        if testloss <= min(testlosses):
            bestmodel = deepcopy(model.state_dict())
            bestmodeldata = [epoch, testloss]
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
            print('time remaining: ' + str(int(((time.time()-t) / progress) * (100 - progress))) + 's')
    print('done in ' + str(round(time.time()-t,2)))
    plt.plot(losses)
    plt.plot(testlosses)
    plt.scatter(bestmodeldata[0], bestmodeldata[1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('epoch = ' + str(n_epochs) + ', lr = ' + str(learning_rate))
    plt.legend(['Training loss', 'Test loss', 'best model at ('+str(bestmodeldata[0])+', '+str(round(bestmodeldata[1], 3))+')'])
    plt.show()
    # load best model
    if best:
        model.load_state_dict(bestmodel)
    return model

def spline(x, start, end):
    return -2*(end - start) * np.power(x, 3) + 3*(end - start) * np.power(x, 2) + start
def nomial(x, start, end, exponent=1.5):
    return (start-end) * np.power(1-x, exponent) + end
def logistic(x, start, end, slope=10, middle=0.5):
    y = (start - end) / (1 + np.power(np.e, -slope * ((1-x) - (1-middle))))
    return y + end

def noise_train_validate(model, loss_fn, optimizer, n_epochs, learning_rate, x_train, y_train, x_test, y_test, best=True, noise=(0, 0), noisedistr=(10, 0.5), testloss_fn=None):
    # slower, but shows test loss to find over fitting and picks best model of all epochs
    from copy import deepcopy
    # extract training data
    x_train = torch.tensor(x_train.iloc[:, :len(x_train.columns)].values)
    x_train = x_train.cuda()
    x_train.requires_grad = True
    y_train = torch.tensor(y_train.iloc[:, -1].values).view(-1, 1)
    y_train = y_train.cuda()

    # extract test / validation data
    x_test = torch.tensor(x_test.iloc[:, :len(x_test.columns)].values)
    x_test = x_test.cuda()
    x_test.requires_grad = True
    y_test = torch.tensor(y_test.iloc[:, -1].values).view(-1, 1)
    y_test = y_test.cuda()

    print('Training starting...')
    losses = []
    testlosses = []
    noiselevels = []
    t = time.time()
    n = 10
    model.train()

    optimizer = optimizer(model.parameters(), lr=learning_rate)
    if testloss_fn == None:
        testloss_fn = loss_fn

    # Train the model
    for epoch in range(n_epochs):
        # noise generation
        x = epoch/n_epochs
        std = logistic(x, noise[0], noise[1], slope=noisedistr[0], middle=noisedistr[1])
        noiselevels.append(std)
        bias = (torch.rand(1) * 2 - 1) * std
        ran = torch.randn(x_train.size()) * std + bias
        # Forward pass
        y_pred_train = model(x_train + ran.cuda())
        y_pred_test = model(x_test)
        # Compute the loss
        if loss_fn == PINNLoss:
            loss = loss_fn(y_pred_train, y_train, x_train)
        else:
            loss = loss_fn(y_pred_train, y_train)
        if testloss_fn == PINNLoss:
            testloss = testloss_fn(y_pred_test, y_test, x_test)
        else:
            testloss = testloss_fn(y_pred_test, y_test)
        losses.append(loss.item())
        testloss = testloss.item()
        testlosses.append(testloss)
        if testloss <= min(testlosses):
            bestmodel = deepcopy(model.state_dict())
            bestmodeldata = [epoch, testloss]
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
            print('time remaining: ' + str(int(((time.time()-t) / progress) * (100 - progress))) + 's')
    print('done in ' + str(round(time.time()-t,2)) + 's')
    plt.plot(losses)
    legend = ['Training loss']
    plt.plot(testlosses)
    legend.append('Test loss')
    if noise != (0,0):
        plt.plot(noiselevels)
        legend.append('Noise level')
    if best:
        plt.scatter(bestmodeldata[0], bestmodeldata[1], c='red')
        legend.append('best model at ('+str(bestmodeldata[0])+', '+str(round(bestmodeldata[1], 3))+')')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('n_epochs = ' + str(n_epochs) + ', lr = ' + str(learning_rate))
    plt.legend(legend)
    plt.ylim(0,1)
    plt.show()
    # load best model
    if best:
        model.load_state_dict(bestmodel)
    return model


def noise_train_validate_animate(model, loss_fn, optimizer, n_epochs, learning_rate, x_train, y_train, x_test, y_test, best=True, testloss_fn=None, noise=(0, 0), noisedistr=(10, 0.5), update_freq=1):
    # slower, but shows test loss to find over fitting and picks best model of all epochs
    from copy import deepcopy
    # extract training data
    x_train = torch.tensor(x_train.iloc[:, :len(x_train.columns)].values)
    x_train = x_train.cuda()
    x_train.requires_grad = True
    y_train = torch.tensor(y_train.iloc[:, -1].values).view(-1, 1)
    y_train = y_train.cuda()

    # extract test / validation data
    x_test = torch.tensor(x_test.iloc[:, :len(x_test.columns)].values)
    x_test = x_test.cuda()
    x_test.requires_grad = True
    y_test = torch.tensor(y_test.iloc[:, -1].values).view(-1, 1)
    y_test = y_test.cuda()

    print('Training starting...')
    losses = []
    testlosses = []
    noiselevels = []
    bestmodeldata = [0, 1000]
    epoch = 0
    t = time.time()
    n = 1/update_freq

    # enable interactive mode
    plt.ion()

    # creating subplot and figure
    fig = plt.figure()
    ax = fig.add_subplot()
    line1, = ax.plot(list(range(epoch)), losses)
    line2, = ax.plot(list(range(epoch)), testlosses)
    line3, = ax.plot(list(range(epoch)), noiselevels)
    line4, = ax.plot(bestmodeldata[0], bestmodeldata[1], 'ro')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Progress: 0%, time remaining:  ...s')
    plt.legend(['Training loss', 'Test loss','Noise level', 'Best model at ('+ str(bestmodeldata[0]) + ', ' + str(bestmodeldata[1])+ ')'])
    plt.ylim(0, 1)
    plt.xlim(0, n_epochs)
    fig.canvas.draw()
    fig.canvas.flush_events()
    c = n
    model.train()
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    if testloss_fn == None:
        testloss_fn = loss_fn
    # Train the model
    for epoch in range(n_epochs):
        # noise generation
        x = epoch/n_epochs
        std = logistic(x, noise[0], noise[1], slope=noisedistr[0], middle=noisedistr[1])
        noiselevels.append(std)
        line3.set_xdata(list(range(epoch+1)))
        line3.set_ydata(noiselevels)
        bias = (torch.rand(1) * 2 - 1) * std
        ran = torch.randn(x_train.size()) * std + bias
        # Forward pass
        y_pred_train = model(x_train + ran.cuda())
        y_pred_test = model(x_test)
        # Compute the loss
        if loss_fn == PINNLoss:
            loss = loss_fn(y_pred_train, y_train, x_train)
        else:
            loss = loss_fn(y_pred_train, y_train)
        if testloss_fn == PINNLoss:
            testloss = testloss_fn(y_pred_test, y_test, x_test)
        else:
            testloss = testloss_fn(y_pred_test, y_test)
        losses.append(loss.item())
        line1.set_xdata(list(range(epoch+1)))
        line1.set_ydata(losses)
        testloss = testloss.item()
        testlosses.append(testloss)
        line2.set_xdata(list(range(epoch+1)))
        line2.set_ydata(testlosses)
        if testloss < bestmodeldata[1]:
            bestmodel = deepcopy(model.state_dict())
            bestmodeldata = [epoch, testloss]
            line4.set_xdata(bestmodeldata[0])
            line4.set_ydata(bestmodeldata[1])
        # Zero the gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # show progress

        elapsed = time.time() - t
        if elapsed - c > 0:
            c += n
            progress = round((epoch/n_epochs) * 100, 1)
            remaining = (elapsed / (epoch+1)) * (n_epochs - epoch)
            plt.title('Progress: '+str(progress)+'%, time remaining: ' + str(round(remaining,2)) + 's')
            plt.legend(['Training loss', 'Test loss', 'Noise level', 'Best model at ('+ str(bestmodeldata[0]) + ', ' + str(round(bestmodeldata[1],3))+ ')'])
            fig.canvas.draw()
            fig.canvas.flush_events()

    plt.title('Progress: 100%, done in: ' + str(round(time.time()-t,2)) + 's')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.ioff()
    input('done in: ' + str(round(time.time()-t,2)) + 's, press enter to continue')
    plt.close(fig)

    # load best model
    if best:
        model.load_state_dict(bestmodel)
    return model