import pandas as pd
import numpy as np
import time
import DPfunctions as dp


file = 'Data/optimatforpy.csv'
saveresult = True

dfbase = pd.read_csv(file)
dfbase = dfbase.set_index('nr')


dp.dfinfo(dfbase)

collums_to_include = [
    'taverage', 'waverage', 'Lnominal', 'Test type',
    'Temp.', 'Fibre Volume Fraction', 'R-value1',
    'Ffatigue', 'Ncycles', 'smax', 'f', 'Laminate',
    'Eit', 'Eic', 'Cut angle '
]
dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Test type', 'exclude_nan')
dp.dfinfo(dfnew)
dfnew = dp.row_filter(dfnew, 'Test type', ['CA'], 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Ncycles', 'exclude_nan')
dp.dfinfo(dfnew)
dfnew = dp.row_filter(dfnew, 'Laminate', ['UD1', 'UD2', 'UD3', 'UD4'], 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Temp.', 'avg')
dp.dfinfo(dfnew)
dfnew = dp.row_filter(dfnew, 'Fibre Volume Fraction', ['#N/A'], 'exclude')
dp.dfinfo(dfnew)

for i in dfnew.index:
    a = float(dfnew.loc[i, 'Eit'])
    b = float(dfnew.loc[i, 'Eic'])
    if pd.isna(a) == True:
        a = 0
    if pd.isna(a) == True:
        b = 0
    c = a + b
    if c > a and c > b:
        c = c/2
    elif c == 0:
        c = np.nan
    dfnew.loc[i, 'Eit'] = c
dp.dfinfo(dfnew)

dfnew = dp.col_filter(dfnew,['Test type', 'Laminate', 'Eic'], 'exclude')
dp.dfinfo(dfnew)
dfnew = dfnew.rename(columns={'Eit': 'E'})
dfnew = dp.cleanup(dfnew, 'E', 'avg')
dp.dfinfo(dfnew)


for i in dfnew.columns:
    b = len(dfnew.index)
    dfnew = dp.cleanup(dfnew, i, 'exclude')
    if len(dfnew.index) / b < 0.95:
        print(i + ' had a lot of nan')
    dfnew[i] = dfnew[i].astype(dtype=float)

print(dfnew.dtypes)
dp.dfinfo(dfnew)

name = 'testdata'
name = name + dp.timetag() + '.csv'

if saveresult:
    pd.DataFrame.to_csv(dfnew, 'processed\\' + name)
    print('File saved as: ' + name)
