import pandas as pd
import numpy as np
import time
import DPfunctions as dp


file = 'Data/optimatforpy.csv'
saveresult = False
tag = 'onlystatic'

print('initial data from ' + file)
dfbase = pd.read_csv(file)
dfbase = dfbase.set_index('nr')
dp.dfinfo(dfbase)
print()
print('DataProcessing:')
collums_to_include = [
    'taverage', 'waverage', 'Lnominal', 'Test type',
    'Fibre Volume Fraction',
    'smax', 'Laminate', 'Environment', 'Lab',
    'Eit', 'Eic', 'Cut angle ', 'area', 'Fmax',
    'runout', 'f', 'R-value1', 'Temp.', 'Ncycles'
]
dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Test type', 'exclude_nan')
#temp column
if 'Temp.' in dfnew.columns:
    for i in dfnew.index:
        t = dfnew.loc[i, 'Temp.']
        if not pd.isna(t):
            try:
                t = float(t)
            except:
                t = t[1::]
                try:
                    t = float(t)
                except:
                    dfnew.loc[i, 'Temp.'] = np.nan
                else:
                    dfnew.loc[i, 'Temp.'] = t
            else:
                dfnew.loc[i, 'Temp.'] = t
    tempdict = {}
    for i in dfnew.index:
        env = dfnew.loc[i, 'Environment'] + dfnew.loc[i, 'Lab']
        t = dfnew.loc[i, 'Temp.']
        if env not in tempdict:
            tempdict[env] = []
        if not pd.isna(t):
            tempdict[env].append(t)
    for i in tempdict:
        if tempdict[i] == []:
            tempdict[i] = np.nan
        else:
            tempdict[i] = np.mean(np.array(tempdict[i]))
    for i in dfnew.index:
        t = dfnew.loc[i, 'Temp.']
        env = dfnew.loc[i, 'Environment'] + dfnew.loc[i, 'Lab']
        if pd.isna(t):
            dfnew.loc[i, 'Temp.'] = tempdict[env]
    #end of temp processing

dfnew = dp.row_filter(dfnew, 'Test type', ['CA'], 'include')
dp.dfinfo(dfnew)
dfnew = dp.row_filter(dfnew, 'Laminate', ['MD2'], 'include')
dp.dfinfo(dfnew)
if 'Ncycles' in dfnew.columns:
    dfnew = dp.cleanup(dfnew, 'Ncycles', 'exclude_nan')
    dfnew = dp.row_filter(dfnew, 'Ncycles', [0.0, 0], 'exclude')
    dp.dfinfo(dfnew)
if 'runout' in dfnew.columns:
    dfnew = dp.row_filter(dfnew, 'runout', ['y', 'Y'], 'exclude')
    dp.dfinfo(dfnew)

# E column
if 'Eit' in dfnew.columns and 'Eic' in dfnew.columns:
    for i in dfnew.index:
        a = float(dfnew.loc[i, 'Eit'])
        b = float(dfnew.loc[i, 'Eic'])
        if pd.isna(a) and pd.isna(b):
            c = np.nan
        elif pd.isna(a) and not pd.isna(b):
            c = b
        elif not pd.isna(a) and pd.isna(b):
            c = a
        else:
            c = (a+b)/2
        dfnew.loc[i, 'Eit'] = c
    dfnew = dfnew.rename(columns={'Eit': 'E'})
    #dfnew = dp.cleanup(dfnew, 'E', 'avg')
    dp.dfinfo(dfnew)



# static test processing
statandfatigue = False
if statandfatigue:
    for i in dfnew.index:
        if dfnew.loc[i, 'Test type'] == 'STT' or dfnew.loc[i, 'Test type'] == 'STC':
            dfnew.loc[i, 'R-value1'] = 1.0
            dfnew.loc[i, 'f'] = 0.0


# end of rvalue
collums_to_exclude = ['Test type', 'Laminate', 'Eic', 'Environment', 'Lab', 'runout']
dfnew = dp.col_filter(dfnew, collums_to_exclude, 'exclude')
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
if 'Ncycles' in dfnew.columns:
    dfnew['Ncycles'] = np.log10(dfnew['Ncycles'])
print()
print('Final stats')
print(dfnew.dtypes)
dp.dfinfo(dfnew)

name = 'data'
if tag == '':
    tag = dp.timetag()
name = name + tag + '.csv'

print()
if saveresult:
    pd.DataFrame.to_csv(dfnew, 'processed\\' + name)
    print('File saved as: ' + name)
else:
    print('results not saved')