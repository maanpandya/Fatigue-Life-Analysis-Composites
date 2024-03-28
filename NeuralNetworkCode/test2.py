import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
from PINNLoss import PINNLoss

path = 'NeuralNetworkCode/NNModelArchive/rev2/mirkodisplaytest11'
model, scaler = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')


#f.test_model(model, scaler, x_test, y_test)

#sn curve
f.sncurvetest(model,20,1,scaler)

x_train = dp.dfread(path + '/x_train.csv')

def spline(x, start, end):
    y = -2*(end - start) * np.power(x, 3) + 3*(end - start) * np.power(x, 2) + start
    return y
def nomial(x, start, exponent):
    y = start * np.power(1-x, exponent)
    return y
def idk(x, start, end, slope):
    s = slope
    t = start
    n = end
    a = -24*n + 16*s + 24*t
    b = 0.5*(105*n - 72*s - 105*t)
    c = -35*n + 24*s + 35*t
    d = 0.5*(15*n - 8*s - 15*t)
    e = 0
    f = t
    np.power(x, 5)
    y = a*np.power(x, 5) + b*np.power(x, 4) + c*np.power(x, 3) + d*np.power(x, 2) + e*x + f
    return y

def idk2(x, start, end, slope):
    s = slope
    t = start
    n = end
    a = -24*n + 16*s + 24*t
    b = 52*n - 40*s - 68*t + 8
    c = -34*n + 32*s + 66*t - 16
    d = 7*n - 8*s - 23*t + 8
    e = 0
    f = t
    np.power(x, 5)
    y = a*np.power(x, 5) + b*np.power(x, 4) + c*np.power(x, 3) + d*np.power(x, 2) + e*x + f
    return y

def logistic(x, start, end, slope=10, middle=0.5):
    y = (start - end) / (1 + np.power(np.e, -slope * ((1-x) - (1-middle))))
    return y + end

'''x = np.linspace(0, 1, 100)

#plt.plot(x, nomial(x, 1, 0.5))
#plt.plot(x, spline(x, 1, 0))
#plt.plot(x, logistic(x, 1, 0, 5, middle=0.4))
plt.plot(x, x)
plt.plot(x, x-1)
#plt.xlim(0,1)
#plt.ylim(0,1)
plt.show()'''