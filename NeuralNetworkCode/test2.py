import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
from PINNLoss import PINNLoss

path = 'NNModelArchive/rev2/0449mse4mrenoisetrained'
model, scaler = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')


#f.test_model(model, scaler, x_test, y_test)

#sn curve
f.sncurvetest(model,5,1,scaler)

x_train = dp.dfread(path + '/x_train.csv')

def spline(x, start, end):
    y = -2*(end - start) * np.power(x, 3) + 3*(end - start) * np.power(x, 2) + start
    return y
def nomial(x, start, exponent):
    y = start * np.power(1-x, exponent)
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