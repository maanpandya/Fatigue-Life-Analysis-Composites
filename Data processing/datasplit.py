import pandas as pd
import numpy as np
import time
import DPfunctions as dp

saveresult = True
file = 'data2.csv'
print('initial data from ' + file)
tag = file[4:-4]
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
print('saving as: ' + trainfile + ', ' + testfile)
if saveresult:
    pd.DataFrame.to_csv(dftrain, 'processed\\' + trainfile)
    pd.DataFrame.to_csv(dftest, 'processed\\' + testfile)
    print('saved')
else:
    print('not saved')