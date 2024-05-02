import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from customloss import PINNLoss
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


def create_model_final(n_inputs, layer_sizes, n_outputs, n_hidden_layers, act_fn, dropout_prob=0.0):
    if type(layer_sizes) != list:
        layer_sizes = n_hidden_layers * [layer_sizes]
    if type(act_fn) != list:
        act_fn = n_hidden_layers * [act_fn]

    # Define a list to hold the layers
    layers = []

    # Add input layer
    layers.append(nn.Linear(n_inputs, layer_sizes[0]))
    layers.append(act_fn[0])
    if dropout_prob > 0.0:
        layers.append(nn.Dropout(dropout_prob))

    # Add hidden layers
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(act_fn[i + 1])
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(dropout_prob))

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

def sncurvetest(model, maxstressratio, dataindex, scalers, orig_data, exportdata=False):
    data = orig_data
    data = data.loc[dataindex]
    data = data.to_frame().T
    print(data)
    data = data.drop(columns=['Ncycles'])
    if 'smax' in data.columns:
        smax_for_nn = True
        smax = data['smax']
    else:
        # calc smax if smax col doesnt exist
        smax_for_nn = False
        smax = ((data['Fmax']* 10**3) / (data['taverage'] * data['waverage'] * 10**-6))*10**-6
    smax = smax.values[0]
    smax_sign = smax >= 0
    data['smax']=0.0

    #Let x be a dataframe with the same columns as data but empty
    x = pd.DataFrame(columns=data.columns)
    #Keep increasing smax from 0 to the initial smax and appending the data to x
    # if smax is negative, do everything in negative numbers
    iterations = np.abs(math.ceil(smax*maxstressratio))
    iterations = 800
    for i in range(iterations):
        i = i
        data['smax'] = float(i)
        # if smax_sign:
        #     data['smax'] = float(i)
        # else:
        #     data['smax'] = float(-i)
        #Append the data to the dataframe x as a row
        x = pd.concat([x, data])
        x['smax'] = x['smax'].astype(float)
    # adjust fmax according to smax
    if 'Fmax' in x.columns:
        x['Fmax'] = x['smax'] * (data['taverage'] * data['waverage']) * 10**-3
    xorig = x.copy()
    # remove smax if it wasnt in the original set
    if not smax_for_nn:
        x = x.drop(columns=['smax'])

    #print(x)

    #Scale x using the values in scalers
    for i in x.columns:
        x[i] = (x[i] - scalers[i]['mean']) / scalers[i]['std']

    #print(xorig)
    #Predict the number of cycles
    model.eval()
    #print dtype of x
    try:
        x = torch.tensor(x.values)
    except:
        print(x)
        raise Exception(x.dtypes)
    x = x.cuda()
    x.requires_grad = True
    y = model(x)
    #Unscale the predicted number of cycles
    y = y.cpu().detach().numpy()
    y = y * scalers['Ncycles']['std'] + scalers['Ncycles']['mean']
    if exportdata:
        return y, xorig['smax']
    else:
        #Plot the results, the column smax from x on the y axis and the predicted number of cycles on the x axis
        plt.scatter(y, xorig['smax'])
        plt.xlabel('log of Ncycles')
        plt.ylabel('smax')
        #Set domain and range of the plot
        #Domain should be more than 0 and less than the maximum value of the predicted number of cycles
        #Range should be more than 0 and less than the maximum value of smax
        plt.xlim(0, 10)
        #plt.ylim(0, iterations)
        plt.show()

def sncurvereal(data, R, export_data=False):
    df = data.loc[data['R-value1'] == float(R)]
    if 'Cut angle ' in df.columns:
        df = df.loc[df['Cut angle '] == 0.0]
    if 'smax' in df.columns:
        s = df['smax']
    else:
        s = (df['Fmax'] * 10**3) / (df['taverage'] * df['waverage'])
    n = df['Ncycles']
    if export_data:
        return s, n
    else:
        plt.title(f'R = {R}')
        plt.scatter(n, s)
        #plt.show()


def sncurvereal2(data, i, err=5, export_data=False):
    datapoint = data.loc[i]
    datapoint = datapoint.to_frame().T
    if 'smax' not in data.columns:
        data['smax'] = (data['Fmax'] * 10 ** 3) / (data['taverage'] * data['waverage'])
    df = dp.find_similar(datapoint, data,
                         ['R-value1', 'Cut angle ', 'taverage', 'waverage'],
                         [], max_error_percent=err)
    indexes = df['indexlists'].to_list()[0] + [i]
    df = data.loc[indexes]
    s = df['smax']
    n = df['Ncycles']
    if export_data:
        return s, n
    else:
        plt.scatter(n, s)

def sncurverealbasic(data):
    s = data['smax']
    n = data['Ncycles']
    plt.scatter(n, s)

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

class fn_01:
    def __repr__(s):
        x = np.linspace(0, 1, 1000)
        plt.plot(x, s.fn(x))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show(block=False)
        plt.pause(3)
        return 'Plotted function...'

class linear(fn_01):
    def __init__(s, start=1, end=0):
        s.b = start
        s.a = end - start
    def fn(s, x):
        return s.a * x + s.b

class spline(fn_01):
    def __init__(s, start=1, end=0):
        s.start=start
        s.end=end
    def fn(s,x):
        return -2 * (s.end - s.start) * np.power(x, 3) + 3 * (s.end - s.start) * np.power(x, 2) + s.start

class nomial(fn_01):
    def __init__(s, start=1, end=0, exponent=2):
        s.start=start - end
        s.end=end
        s.exp=exponent
    def fn(s, x):
        return s.end + s.start * np.power(1-x, s.exp)

class logistic(fn_01):
    def __init__(s, start=1, end=0, slope=10, middle=0.5):
        s.target_range = [start, end]
        s.m = 1 - middle
        s.sl = -slope
        s.range = [1 / (1 + np.power(np.e, s.sl * (1 - s.m))), 1 / (1 + np.power(np.e, s.sl * -s.m))]
        s.target_range.sort(reverse=True)
        s.range.sort(reverse=True)
    def fn(s, x):
        y = 1 / (1 + np.power(np.e, s.sl * ((1-x) - s.m)))
        y = (y - s.range[1]) / (s.range[0] - s.range[1])
        y = y * (s.target_range[0] - s.target_range[1]) + s.target_range[1]
        return y

class n_to_x(fn_01):
    def __init__(self, base, xfactor):
        self.a = xfactor
        self.b = base
    def fn(self, x):
        return self.a * x * np.power(self.b, x * self.a)

class wave(fn_01):
    def __init__(s, amp=0.5, min=0, freq=0.5):
        s.amp = amp
        s.b = min
        s.f = freq
        s.shift = -0.25 / freq
    def fn(s, x):
        return s.amp * np.sin((x - s.shift)*s.f*2*np.pi) + s.amp + s.b

class variable_top_wave(fn_01):
    def __init__(s, topfn=linear(1,0), min=0, freq=10):
        s.ampfn = topfn
        s.b = min
        s.f = freq
        s.shift = -0.25 / freq
    def fn(s, x):
        amp = s.ampfn.fn(x) / 2 - s.b/2
        return amp * np.sin((x - s.shift)*s.f*2*np.pi) + amp + s.b



def train_final(model, loss_fn, optimizer, n_epochs, learning_rate, x_train, y_train,
                x_test=None, y_test=None,
                best=True, testloss_fn=None, noise_fn=None, anti_overfit=False,
                update_freq=1, animate=False, force_no_test=False):

    from copy import deepcopy
    # initialize values
    if testloss_fn == None:
        testloss_fn = loss_fn
    noise = type(noise_fn) != type(None)
    tst = True
    if type(x_test)==type(None) or type(y_test)==type(None) or force_no_test:
        if best==True:
            msg = 'Cannot pick best model without test data.'
            if force_no_test:
                msg = 'force no test == True, so cant pick best model.'
            print(msg + ' best set to False.')
            best = False
        tst = False

    # extract training data
    x_train = torch.tensor(x_train.iloc[:, :len(x_train.columns)].values)
    x_train = x_train.cuda()
    x_train.requires_grad = True
    y_train = torch.tensor(y_train.iloc[:, -1].values).view(-1, 1)
    y_train = y_train.cuda()
    # init some noise variables
    if noise:
        x_train_size = x_train.size()
        y_train_size = y_train.size()

    # extract test / validation data
    if tst:
        x_test = torch.tensor(x_test.iloc[:, :len(x_test.columns)].values)
        x_test = x_test.cuda()
        x_test.requires_grad = True
        y_test = torch.tensor(y_test.iloc[:, -1].values).view(-1, 1)
        y_test = y_test.cuda()

    print('Training starting...')
    losses = []
    if tst:
        testlosses = []
    if noise:
        noiselevels = []
    if best:
        bestmodeldata = [0, np.inf]
    epoch = 0
    t = time.time()
    period = 1 / update_freq
    c = period

    if animate:
        # enable interactive mode
        plt.ion()
        # creating subplot and figure
        fig = plt.figure()
        ax = fig.add_subplot()
        line1, = ax.plot(list(range(epoch)), losses)
        legend = ['Training loss']
        if tst:
            line2, = ax.plot(list(range(epoch)), testlosses)
            legend.append('Test loss')
        if noise:
            line3, = ax.plot(list(range(epoch)), noiselevels)
            legend.append('Noise level')
        if best:
            line4, = ax.plot(bestmodeldata[0], bestmodeldata[1], 'ro')
            legend.append('Best model at ('+ str(bestmodeldata[0]) + ', ' + str(bestmodeldata[1])+ ')')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Progress: 0%, time remaining:  ...s')
        plt.legend(legend)
        plt.ylim(0, 2)
        plt.xlim(0, n_epochs)
        fig.canvas.draw()
        fig.canvas.flush_events()

    model.train()
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(n_epochs):
        # noise generation
        if noise:
            x = epoch/n_epochs
            std = noise_fn.fn(x)
            if epoch > 2 and anti_overfit:
                running_mean_loss = np.mean(losses[-min(epoch, 100):-1])
                running_mean_testloss = np.mean(testlosses[-min(epoch, 100):-1])
                p = 1
                a = 1
                overfit = np.power(max((running_mean_testloss / running_mean_loss) * a, 1), p) - 1
                std = std + overfit
            noiselevels.append(std)
            biasx = (torch.rand(1) * 2 - 1) * std
            ranx = torch.randn(x_train_size) * std + biasx
            x_train_temp = x_train + ranx.cuda()
        else:
            x_train_temp = x_train
        # Forward pass and compute the loss
        y_pred_train = model(x_train_temp)
        if loss_fn == PINNLoss:
            loss = loss_fn(y_pred_train, y_train, x_train, sevencutoff=1.4, indexsmax=6)
        else:
            loss = loss_fn(y_pred_train, y_train)
        losses.append(loss.item())
        # repeat for test data
        if tst:
            y_pred_test = model(x_test)
            if testloss_fn == PINNLoss:
                testloss = testloss_fn(y_pred_test, y_test, x_test, sevencutoff=1.9, indexsmax=6)
            else:
                testloss = testloss_fn(y_pred_test, y_test)
            testloss = testloss.item()
            testlosses.append(testloss)
            if best:
                if testloss < bestmodeldata[1]:
                    bestmodel = deepcopy(model.state_dict())
                    bestmodeldata = [epoch, testloss]
        # Zero the gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # show progress

        elapsed = time.time() - t
        if elapsed - c > 0 or epoch == n_epochs-2:
            c += period
            progress = round((epoch/n_epochs) * 100, 1)
            remaining = (elapsed / (epoch+1)) * (n_epochs - epoch)
            if animate:
                legend = ['Training loss']
                line1.set_xdata(list(range(epoch + 1)))
                line1.set_ydata(losses)
                if tst:
                    line2.set_xdata(list(range(epoch + 1)))
                    line2.set_ydata(testlosses)
                    legend.append('Test loss')
                if noise:
                    line3.set_xdata(list(range(epoch + 1)))
                    line3.set_ydata(noiselevels)
                    legend.append('Noise level')
                if best:
                    line4.set_xdata(bestmodeldata[0])
                    line4.set_ydata(bestmodeldata[1])
                    legend.append('Best model at ('+ str(bestmodeldata[0]) + ', ' + str(round(bestmodeldata[1],3))+ ')')
                plt.title('Progress: '+str(progress)+'%, time remaining: ' + str(round(remaining,2)) + 's')
                plt.legend(legend)
                fig.canvas.draw()
                fig.canvas.flush_events()
            msg = 'Progress: ' + str(progress) + '%, time remaining: ' + str(round(remaining, 2)) + 's'
            print('\r'+msg, end='                             ')

    tot_time = round(time.time() - t, 2)
    print()
    print('done in: ' + str(tot_time) + 's, at ' + str(int(n_epochs/tot_time)) + ' epochs/s.')
    if animate:
        plt.title('Progress: 100%, done in: ' + str(tot_time) + 's')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)
        plt.close(fig)
        plt.ioff()
    else:
        plt.plot(losses)
        legend = ['Training loss']
        if tst:
            plt.plot(testlosses)
            legend.append('Test loss')
        if noise:
            plt.plot(noiselevels)
            legend.append('Noise level')
        if best:
            plt.scatter(bestmodeldata[0], bestmodeldata[1], c='red')
            legend.append('best model at (' + str(bestmodeldata[0]) + ', ' + str(round(bestmodeldata[1], 3)) + ')')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('n_epochs = ' + str(n_epochs) + ', lr = ' + str(learning_rate))
        plt.legend(legend)
        plt.ylim(0, 2)
        plt.show()
    # load best model
    if best:
        model.load_state_dict(bestmodel)
    return model