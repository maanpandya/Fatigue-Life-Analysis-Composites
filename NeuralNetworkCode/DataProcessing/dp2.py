import pandas as pd
import numpy as np
import DPfunctions as dp


cols = ['Ncycles', 'smax', 'smean', 'smin', 'Lnominal', 'taverage', 'waverage', 'area']
#laminates = ['UD1', 'UD2', 'UD3', 'UD4', 'UD5']
laminates = ['MD2']
geometries = []#['R0400', 'R0500', 'R0300', 'D0200', 'D1200', 'I0100', 'I0200']
tests = ['CA', 'STT', 'STC']
tag = 'noR10'
save = False
split_on_R, R, use_below = True, 10, True
absmax = False
correct_smax = True


collums_to_include = [
    'Lnominal', 'taverage', 'waverage', 'area',
    'Test type', 'Laminate', 'Cut angle ',
    'R-value1', 'Ncycles', 'smax', 'Fmax'
    ,'runout', 'Geometry'
]

dfbase = pd.read_csv('Data/optimatforpy.csv')
dfbase = dfbase.set_index('nr')
dp.dfinfo(dfbase, 'base')

dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
dp.dfinfo(dfnew, 'col selection')

dfnew = dp.row_filter(dfnew, 'Laminate', laminates, 'include')
dp.dfinfo(dfnew, 'filter on laminate')
if len(geometries) > 0:
    dfnew = dp.row_filter_remove(dfnew, 'Geometry', geometries, 'include')
    dp.dfinfo(dfnew, 'filter on geometry')
else:
    dfnew = dp.col_filter(dfnew, ['Geometry'], 'exclude')

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

if split_on_R:
    dfup, dfdown = dp.filter_dataframe_by_cutoff(dfnew, 'R-value1', R)
    if use_below:
        dfnew = dfdown
    else:
        dfnew = dfup

if correct_smax:
    dfnew['smean'] = dp.rmath({'smax':dfnew['smax'], 'R':dfnew['R-value1']}, 'smean')
    for i in dfnew.index:
        smax = dfnew['smax'].loc[i]
        R = dfnew['R-value1'].loc[i]
        if smax < 0 and R != 0:
            dfnew.loc[i, 'smax'] = smax/R
            dfnew.loc[i, 'smean'] = dp.rmath({'smax':dfnew.loc[i, 'smax'], 'R':dfnew.loc[i, 'R-value1']}, 'smean')
        elif smax < 0 and R == 0:
            dfnew.loc[i, 'smax'] = 0
            dfnew.loc[i, 'smean'] = smax/2
    dfnew['smin'] = dp.rmath({'smax':dfnew['smax'], 'smean':dfnew['smean']}, 'smin')
else:
    dfnew['smean'] = ((1+dfnew['R-value1']) / 2) * dfnew['smax']
    for i in dfnew.index:
        smax = dfnew['smax'].loc[i]
        R = dfnew['R-value1'].loc[i]
        if smax < 0 and R != 0:
            dfnew.loc[i, 'smean'] = (smax / 2)*(1 + (1 / R))


    if absmax:
        dfnew['smax'] = np.abs(dfnew['smax'])
        dfnew['Fmax'] = np.abs(dfnew['Fmax'])
        dfnew['smean'] = np.abs(dfnew['smean'])

print(dfnew['R-value1'].max())
dfnew = dp.col_filter(dfnew, cols, 'include')
if 'Ncycles' in dfnew.columns:
    dfnew['Ncycles'] = np.log10(dfnew['Ncycles'])
dp.dfinfo(dfnew, 'final')
name = 'data'
if tag == '':
    tag = dp.timetag()
name = name + tag + '.csv'
print()
if save:
    dfnew.to_csv('processed/' + name)
    print('File saved as: ' + name)
else:
    print('results not saved')