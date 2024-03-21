import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
from PINNLoss import PINNLoss

path = 'NNModelArchive/rev2/10x30pinloss'
model, scaler = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')

f.test_model(model, scaler, x_test, y_test)

#sn curve
f.sncurvetest(model,1,scaler)
