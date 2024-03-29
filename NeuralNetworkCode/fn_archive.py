import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from PINNLoss import PINNLoss
import DataProcessing.DPfunctions as dp
import os
import pickle
import math

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

