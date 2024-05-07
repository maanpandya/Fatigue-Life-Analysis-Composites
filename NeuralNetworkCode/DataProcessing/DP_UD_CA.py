import pandas as pd
import numpy as np
import DPfunctions as dp


cols = ['Ncycles', 'smax', 'smean', 'Lnominal', 'taverage', 'waverage', 'area', 'Fmax', 'R-value1']
#laminates = ['UD1', 'UD2', 'UD3', 'UD4', 'UD5']
laminates = ['MD2']
tests = ['CA', 'STT', 'STC']
tag = '10'
save = True


collums_to_include = [
    'Lnominal', 'taverage', 'waverage', 'area',
    'Test type', 'Laminate', 'Cut angle ',
    'R-value1', 'Ncycles', 'smax', 'Fmax'
    ,'runout'
]

dfbase = pd.read_csv('Data/optimatforpy.csv')
dfbase = dfbase.set_index('nr')
dp.dfinfo(dfbase, 'base')

dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
dp.dfinfo(dfnew, 'col selection')

dfnew = dp.row_filter(dfnew, 'Laminate', laminates, 'include')
dp.dfinfo(dfnew, 'filter on laminate')

dfnew = dp.row_filter_remove(dfnew, 'Test type', tests, 'include')

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
for i in dfnew.index:
    smax = dfnew['smax'].loc[i]
    R = dfnew['R-value1'].loc[i]
    if smax < 0 and R != 0:
        dfnew.loc[i, 'smean'] = (smax / 2)*(1 + (1 / R))
dfnew = dp.col_filter(dfnew, cols, 'include')
if 'Ncycles' in dfnew.columns:
    dfnew['Ncycles'] = np.log10(dfnew['Ncycles'])
dp.dfinfo(dfnew, 'final')

dfup, dfdown = dp.filter_dataframe_by_cutoff(dfnew, 'smax', 0)
#dfnew = dfdown
dp.dfinfo(dfnew, 'final final')
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