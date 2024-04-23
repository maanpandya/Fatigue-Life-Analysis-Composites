import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
from PINNLoss import PINNLoss
import time


path = 'NNModelArchive/rev2/0449mse4mrenoisetrained'
model, scaler = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')


#f.test_model(model, scaler, x_test, y_test)

#sn curve
#f.sncurvetest(model,5,1,scaler)

x_train = dp.dfread(path + '/x_train.csv')


class spline:
    def __init__(s, start, end):
        s.start=start
        s.end=end
    def fn(s,x):
        return -2 * (s.end - s.start) * np.power(x, 3) + 3 * (s.end - s.start) * np.power(x, 2) + s.start

class nomial:
    def __init__(s, start, end, exponent):
        s.start=start - end
        s.end=end
        s.exp=exponent
    def fn(s, x):
        return s.end + s.start * np.power(1-x, s.exp)

class logistic:
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



def logisticx(x, start, end, slope=15, middle=0.5):
    y0 = 1 / (1 + np.power(np.e, -slope * ((1-0) - (1-middle))))
    y1 = 1 / (1 + np.power(np.e, -slope * ((1-1) - (1-middle))))
    y = 1 / (1 + np.power(np.e, -slope * ((1-x) - (1-middle))))
    return y

x = np.linspace(0, 1, 100)


plt.plot(x, logistic(0.5, 1, -10, 0.5).fn(x))

plt.xlim(0,1)
plt.ylim(0,1)
plt.show()