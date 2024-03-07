import pandas as pd
import numpy as np
import time
import DPfunctions as dp


file = 'Data/optimatforpy.csv'
saveresult = False
filldata = True

print('initial data from ' + file)
dfbase = pd.read_csv(file)
dfbase = dfbase.set_index('nr')
dp.dfinfo(dfbase)
print()
print('Data processing:')
collums_to_include = [
    'taverage', 'waverage', 'Lnominal', 'Test type',
    'Temp.', 'Fibre Volume Fraction', 'R-value1',
    'smax', 'f', 'Laminate',
    'Eit', 'Eic', 'Cut angle ', 'Ncycles'
]
dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Test type', 'exclude_nan')
dfnew = dp.row_filter(dfnew, 'Test type', ['CA'], 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Ncycles', 'exclude_nan')
dfnew = dp.row_filter(dfnew, 'Ncycles', [0.0, 0], 'exclude')
dp.dfinfo(dfnew)
dfnew = dp.row_filter(dfnew, 'Laminate', ['UD1', 'UD2', 'UD3', 'UD4'], 'include')
dp.dfinfo(dfnew)

# E column
for i in dfnew.index:
    a = float(dfnew.loc[i, 'Eit'])
    b = float(dfnew.loc[i, 'Eic'])
    if pd.isna(a) == True:
        a = 0
    if pd.isna(b) == True:
        b = 0
    c = a + b
    if c > a and c > b:
        c = c/2
    elif c == 0:
        c = np.nan
    dfnew.loc[i, 'Eit'] = c
dp.dfinfo(dfnew)
dfnew = dfnew.rename(columns={'Eit': 'E'})
if filldata:
    dfnew = dp.cleanup(dfnew, 'E', 'avg')
    dp.dfinfo(dfnew)
    dfnew = dp.cleanup(dfnew, 'Temp.', 'avg')
    dp.dfinfo(dfnew)

dfnew = dp.col_filter(dfnew,['Test type', 'Laminate', 'Eic'], 'exclude')
dp.dfinfo(dfnew)
print()
print('Final cleanup')
for i in dfnew.columns:
    b = len(dfnew.index)
    dfnew = dp.cleanup(dfnew, i, 'exclude')
    if len(dfnew.index) / b < 1:
        print(i + ': ' + str(b - len(dfnew.index)) + '/' + str(b) + ' removed')
    dfnew[i] = dfnew[i].astype(dtype=float)

# log of some columns
dfnew['Ncycles'] = np.log10(dfnew['Ncycles'])
print()
print('Final stats')
print(dfnew.dtypes)
dp.dfinfo(dfnew)

name = 'testdata'
tag = dp.timetag()
name = name + tag + '.csv'

if saveresult:
    pd.DataFrame.to_csv(dfnew, 'processed\\' + name)
    print()
    print('File saved as: ' + name)