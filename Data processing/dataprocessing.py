import pandas as pd
import numpy as np
import time
import DPfunctions as dp


file = 'Data/optimatforpy.csv'
saveresult = False
tag = '3'

print('initial data from ' + file)
dfbase = pd.read_csv(file)
dfbase = dfbase.set_index('nr')
dp.dfinfo(dfbase)
print()
print('Data processing:')
collums_to_include = [
    'taverage', 'waverage', 'Lnominal', 'Test type',
    'Temp.', 'Fibre Volume Fraction', 'R-value1',
    'smax', 'f', 'Laminate', 'Environment', 'Lab',
    'Eit', 'Eic', 'Cut angle ', 'Ncycles', 'area', 'Ffatigue'
]
dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Test type', 'exclude_nan')
dfnew = dp.row_filter(dfnew, 'Test type', ['CA'], 'include')
dp.dfinfo(dfnew)
dfnew = dp.row_filter(dfnew, 'Laminate', ['UD1', 'UD2', 'UD3', 'UD4', 'UD5'], 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Ncycles', 'exclude_nan')
dfnew = dp.row_filter(dfnew, 'Ncycles', [0.0, 0], 'exclude')
dp.dfinfo(dfnew)


# E column
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

#temp column
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
                dfnew = pd.DataFrame.drop(dfnew, i)
                print('dropped a row because temp')
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

collums_to_exclude = ['Test type', 'Laminate', 'Eic', 'Environment', 'Lab']
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