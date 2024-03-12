import pandas as pd
import numpy as np
import time
import DPfunctions as dp

saveresult = True
file = 'data120324101615.csv'
print('initial data from ' + file)
tag = file[-16:-4]
file = 'processed/' + file
dfbase = pd.read_csv(file)
dfbase = dfbase.set_index('nr')
dftrain, dftest = dp.datasplitscale(dfbase, 0.2, ['Ncycles'])
dp.dfinfo(dftrain)
dp.dfinfo(dftest)
trainfile = 'traindata'
trainfile += tag + '.csv'
testfile = 'testdata'
testfile += tag + '.csv'
if saveresult:
    pd.DataFrame.to_csv(dftrain, 'processed\\' + trainfile)
    pd.DataFrame.to_csv(dftest, 'processed\\' + testfile)