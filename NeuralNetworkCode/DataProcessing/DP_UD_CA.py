import pandas as pd
import numpy as np
import DPfunctions as dp


# ,taverage,waverage,area,
cols = ['Ncycles', 'smax', 'smean']
laminates = ['UD1', 'UD2', 'UD3', 'UD4', 'UD5']
tests = ['CA', 'STT', 'STC']
tag = '8'
save = True


collums_to_include = [
    'Lnominal', 'Test type',
    'smax', 'Laminate', 'Cut angle ',
    'R-value1', 'Ncycles'
    ,'runout'
]

dfbase = pd.read_csv('Data/optimatforpy.csv')
dfbase = dfbase.set_index('nr')
dp.dfinfo(dfbase, 'base')

dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
dp.dfinfo(dfnew, 'col selection')

dfnew = dp.row_filter(dfnew, 'Laminate', laminates, 'include')
dp.dfinfo(dfnew, 'filter on laminate')

dfnew = dp.cleanup(dfnew, 'Ncycles', 'exclude_nan')
dfnew = dp.row_filter(dfnew, 'Ncycles', [0.0, 0], 'exclude')
dp.dfinfo(dfnew, 'clean ncycles')

if 'runout' in dfnew.columns:
    dfnew = dp.row_filter_remove(dfnew, 'runout', ['y', 'Y'], 'exclude')
    dp.dfinfo(dfnew, 'remove runout rows')
if 'Cut angle ' in dfnew.columns:
    dfnew = dp.row_filter_remove(dfnew, 'Cut angle ', ['0', '0.0', 0, 0.0], 'include')
    dp.dfinfo(dfnew, 'remove non zero cut angle rows')

dp.label_col(dfnew, 'Laminate', laminates)
dp.dfinfo(dfnew, 'laminate labels')

for i in dfnew.index:
    if pd.isna(dfnew['R-value1'].loc[i]):
        dfnew.loc[i, 'R-value1'] = 0
dfnew = dp.col_filter(dfnew, cols+['R-value1'], 'include')
dfnew = dp.big_cleanup(dfnew)
dfnew['smean'] = ((1+dfnew['R-value1']) / 2) * dfnew['smax']
dfnew = dp.col_filter(dfnew, cols, 'include')
print(dfnew['Ncycles'])
if 'Ncycles' in dfnew.columns:
    dfnew['Ncycles'] = np.log10(dfnew['Ncycles'])
dp.dfinfo(dfnew, 'final')

name = 'data'
if tag == '':
    tag = dp.timetag()
name = name + tag + '.csv'
print()
if save:
    dfnew.to_csv('processed\\' + name)
    print('File saved as: ' + name)
else:
    print('results not saved')