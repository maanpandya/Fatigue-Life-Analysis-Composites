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
import copy

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


def test_model(model, scaler, x_test, y_test, n_is_log=True, plot=True):
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
    if n_is_log:
        pred_eval['pred_log'] = pred_eval['pred_scaled'] * scaler['std'] + scaler['mean']
        pred_eval['real_log'] = pred_eval['real_scaled'] * scaler['std'] + scaler['mean']
        pred_eval['pred'] = np.power(10, pred_eval['pred_log'])
        pred_eval['real'] = np.power(10, pred_eval['real_log'])
    else:
        pred_eval['pred'] = pred_eval['pred_scaled'] * scaler['std'] + scaler['mean']
        pred_eval['real'] = pred_eval['real_scaled'] * scaler['std'] + scaler['mean']
        pred_eval['pred_log'] = np.log10(pred_eval['pred'])
        pred_eval['real_log'] = np.log10(pred_eval['real'])

    # print various measures of accuracy
    error_dict = {
        'lMSE': np.mean(np.power(pred_eval['pred_log'] - pred_eval['real_log'], 2)),
        'lRMSE': np.sqrt(np.mean(np.power(pred_eval['pred_log'] - pred_eval['real_log'], 2))),
        'lMAE': np.mean(np.abs(pred_eval['pred_log'] - pred_eval['real_log'])),
        'MRE': np.mean(np.abs((pred_eval['pred'] - pred_eval['real']) / (pred_eval['real'])))
    }
    print('Measures of error:')
    print(error_dict)
    # outlier detection
    lAE = np.abs(pred_eval['pred_log'] - pred_eval['real_log'])
    lAE = lAE.sort_values(ascending=False)
    lE = pred_eval['pred_log'] - pred_eval['real_log']
    print('top 5 lAE:')
    print(lE.loc[lAE.index[0:5:]])

    # plot errors
    if plot:
        bound1 = error_dict['lMAE']
        bound2 = 1
        plt.plot([-100, 100], [-100, 100], color='red', linestyle='--')
        plt.plot([-100, 100], [-100-bound1, 100-bound1], color='darkred', linestyle='--')
        plt.plot([-100, 100], [-100+bound1, 100+bound1], color='darkred', linestyle='--')
        plt.plot([-100, 100], [-100-bound2, 100-bound2], color='black', linestyle='--')
        plt.plot([-100, 100], [-100+bound2, 100+bound2], color='black', linestyle='--')
        plt.scatter(pred_eval['real_log'], pred_eval['pred_log'])
        plt.xlabel('y_test')
        plt.ylabel('predicted')
        plt.legend()
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    return error_dict

def sncurvetest(model, maxstress, datapoint, scalers, exportdata=False):
    data = datapoint
    data = data.drop(columns=['Ncycles'])
    if 'smax' in data.columns:
        smax_for_nn = True
        smax = data['smax']
    else:
        # calc smax if smax col doesnt exist
        smax_for_nn = False
        smax = ((data['Fmax']* 10**3) / (data['taverage'] * data['waverage'] * 10**-6))*10**-6
    smax = smax.values[0]
    R = data['R-value1'].values[0]
    iscompressive = R >= 1
    data['smax']=0.0

    #Let x be a dataframe with the same columns as data but empty
    x = pd.DataFrame(columns=data.columns)
    #Keep increasing smax from 0 to the initial smax and appending the data to x
    # if smax is negative, do everything in negative numbers
    extra = 2
    iterations = maxstress * extra
    for i in range(iterations):
        i = i/extra
        data['smax'] = float(i)
        if iscompressive:
             data['smean'] = (i/2)*(1+1/R)
        else:
             data['smean'] = (i/2)*(1+R)
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

    y = y * scalers['Ncycles']['std'] + scalers['Ncycles']['mean']
    #gradient test
    gradient1 = torch.autograd.grad(torch.mean(y), x, create_graph=True)[0][:, 6].cpu().detach().numpy()
    S = np.linspace(0, maxstress, num=iterations)
    center = 0
    offset = 0
    amp = 2*10**3
    plt.plot((gradient1+offset) * amp + center, S, label='1st gradient')
    plt.plot(center*S/S, S, linestyle='--', color='grey', label='zero gradient')

    #Unscale the predicted number of cycles
    y = y.cpu().detach().numpy()
    #y = y * scalers['Ncycles']['std'] + scalers['Ncycles']['mean']
    if exportdata:
        return xorig['smax'], y
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

def sncurverealbasic(data, export_data=False):
    s = data['smax']
    n = data['Ncycles']
    if export_data:
        return s, n
    else:
        plt.scatter(n, s)


def complete_sn_curve(model, scaler, data, datapoint):
    if 'smax' not in data.columns:
        data['smax'] = (data['Fmax'] * 10 ** 3) / (data['taverage'] * data['waverage'])
        datapoint['smax'] = (datapoint['Fmax'] * 10 ** 3) / (datapoint['taverage'] * datapoint['waverage'])
    cols = ['taverage', 'waverage', 'Lnominal']
    i = datapoint.index[0]
    statindexes = []
    if 'R-value1' in data.columns:
        cols.append('R-value1')
        statdp = copy.deepcopy(datapoint)
        print(statdp['R-value1'])
        statdp.loc[i, 'R-value1'] = 0
        print(statdp['R-value1'])
        df = dp.find_similar(statdp, data,
                             cols,
                             [], max_error_percent=1)
        statindexes = df['indexlists'].to_list()[0]
        if type(statindexes) != list:
            statindexes = []
    df = dp.find_similar(datapoint, data,
                         cols,
                         [], max_error_percent=2)
    indexes = df['indexlists'].to_list()[0]
    if type(indexes) != list:
        indexes = []
    df = data.loc[indexes + statindexes]
    srs = df['smax']
    nrs = df['Ncycles']
    if 'R-value1' in data.columns:
        R = datapoint['R-value1'].values[0]
        src, nrc = sncurvereal(data, R, export_data=True)
    else:
        max = datapoint['smax'].values[0]
        mean = datapoint['smean'].values[0]
        mini = 2 * mean - max
        r = mini / max
        if max < 0:
            r = 1 / r
        R = r
        src, nrc = sncurverealbasic(data, export_data=True)
    srp, nrp = sncurvetest(model, 800, datapoint, scaler, exportdata=True)
    plt.scatter(nrc, src, color='black', label='All experimental')
    plt.scatter(nrs, srs, color='orange', label='Similar experimental')
    plt.plot(nrp, srp, color='red', label='Model prediction')
    plt.xlim(-2,8)
    plt.title(f'R = {R}')
    plt.xlabel('log(N) [-]')
    plt.ylabel('Maximum stress [MPa]')
    plt.legend()
    plt.show()

def randomcolor(min=0, max=1):
    range = min, max
    color = np.array([np.random.uniform(range[0],range[1]), np.random.uniform(range[0],range[1]), np.random.uniform(range[0],range[1])])
    return color
def invertcolor(color):
    return np.array([1, 1, 1]) - color
def reshade(color, rng=0.1):
    color = color + np.array([np.random.uniform(-rng, rng), np.random.uniform(-rng, rng),
              np.random.uniform(-rng, rng)])
    color = (color<=1) * (color>=0) * color + (color>1)
    return color
def complete_sncurve2(datapoint, data, R, model, scaler, minstress=0, maxstress=800,
                      exp=True, name='', color=None, plot_abs=True, axis=None, unlog_n=False, amp_s=False):
    if type(color)==type(None):
        color = randomcolor()
    expcolor = np.append(color * 0.8, 0.5)
    predcolor = color
    data = copy.deepcopy(data)
    noR = False
    if 'R-value1' not in data.columns:
        noR = True
        data['R-value1'] = dp.rmath({'smean':data['smean'], 'smax':data['smax']}, 'R')
    if exp:
        expdata = data[data['R-value1']==R]
        expn = expdata['Ncycles']
        if R <= 1:
            exps = expdata['smax']
        else:
            exps = expdata['smin']
            if plot_abs:
                exps = -exps
        if unlog_n:
            expn = 10**expn
        if axis == None:
            plt.scatter(expn, exps, label=f'experimental R = {R}', color=expcolor)
        else:
            axis.scatter(expn, exps, label=f'experimental R = {R}', color=expcolor)
    if 'Ncycles' in datapoint.columns:
        datapoint = datapoint.drop(columns=['Ncycles'])
    datapoint['R-value1'] = R
    if R <= 1:
        stressrange = np.arange(minstress, maxstress, 1)
        datapoint['smax'] = minstress
    else:
        stressrange = np.arange(-maxstress/R, -minstress/R, 1/R)
        stressrange = np.flip(stressrange)
        datapoint['smax'] = -minstress/R
    x = copy.deepcopy(datapoint)
    for i in stressrange:
        datapoint['smax'] = float(i)
        x = pd.concat([x, datapoint])
        x['smax'] = x['smax'].astype(float)
    if 'smean' in x.columns:
        x['smean'] = dp.rmath({'smax':x['smax'], 'R':x['R-value1']}, 'smean')
        x['smean'] = x['smean'].astype(float)
    if 'smin' in x.columns:
        x['smin'] = dp.rmath({'smax':x['smax'], 'R':x['R-value1']}, 'smin')
        x['smin'] = x['smin'].astype(float)
    if 'samp' in x.columns:
        x['samp'] = dp.rmath({'smax':x['smax'], 'R':x['R-value1']}, 'samp')
        x['samp'] = x['samp'].astype(float)
    if 'Fmax' in x.columns:
        x['Fmax'] = x['smax'] * (data['taverage'] * data['waverage']) * 10**-3
    if noR:
        x = x.drop(columns=['R-value1'])
    else:
        x['R-value1'] = x['R-value1'].astype(float)
    for i in x.columns:
        x[i] = (x[i] - scaler[i]['mean']) / scaler[i]['std']
    model.eval()
    try:
        x = torch.tensor(x.values)
    except:
        print(x)
        raise Exception(x.dtypes)
    x = x.cuda()
    x.requires_grad = True
    npred = model(x).cpu().detach().numpy()
    npred = npred * scaler['Ncycles']['std'] + scaler['Ncycles']['mean']
    if R <= 1: # smax is plotted
        smax = np.insert(stressrange, 0, minstress)
        if amp_s: # plot s amp
            samp = dp.rmath({'R': R, 'smax': smax}, 'samp')
            spred = samp
        else: # plot smax
            spred = smax
    else: # smin is plotted
        smax = np.insert(stressrange, 0, -minstress/R)
        if amp_s:
            samp = dp.rmath({'R': R, 'smax': smax}, 'samp')
            spred = samp
        else:
            smin = dp.rmath({'R':R, 'smax':smax}, 'smin')
            spred = smin
        if plot_abs:
            spred = np.abs(spred)
    if unlog_n:
        npred = 10**npred
    if axis == None:
        plt.plot(npred, spred, label=f'R = {R}, pred by {name}', color=predcolor)
    else:
        axis.plot(npred, spred, label=f'R = {R}, pred by {name}', color=predcolor)



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
        amp = amp * (amp>0)
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
    y_train.requires_grad = True
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
        y_test.requires_grad = True

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
            legend.append('Validation loss')
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
            loss = loss_fn(y_pred_train, y_train, x_train, indexsmax=6)
        else:
            loss = loss_fn(y_pred_train, y_train)
        losses.append(loss.item())
        # repeat for test data
        if tst:
            y_pred_test = model(x_test)
            if testloss_fn == PINNLoss:
                testloss = testloss_fn(y_pred_test, y_test, x_test, indexsmax=6)
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
                legend = [f'Training loss = {round(losses[-1],3)}']
                line1.set_xdata(list(range(epoch + 1)))
                line1.set_ydata(losses)
                if tst:
                    line2.set_xdata(list(range(epoch + 1)))
                    line2.set_ydata(testlosses)
                    legend.append(f'Validation loss = {round(testlosses[-1],3)}')
                if noise:
                    line3.set_xdata(list(range(epoch + 1)))
                    line3.set_ydata(noiselevels)
                    legend.append(f'Noise level = {round(noiselevels[-1],3)}')
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
            legend.append('Validation loss')
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