import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DataProcessing.DPfunctions as dp
import function as f
import time


path = 'NNModelArchive/good/sinnoisefn0375'
model, scaler = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')

f.test_model(model, scaler, x_test, y_test)

#sn curve
#f.sncurvetest(model,11,10,scaler, testdatafile=test)

x_train = dp.dfread(path + '/x_train.csv')

