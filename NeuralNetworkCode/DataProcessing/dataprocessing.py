import pandas as pd
import numpy as np
import time
import DPfunctions as dp


file = 'Data/optimatforpy.csv'
saveresult = False
stat_ref = False
statandfatigue = False
no_constant_cols = True
no_related_cols = False
tag = '7'

print('initial data from ' + file)
dfbase = pd.read_csv(file)
dfbase = dfbase.set_index('nr')
dp.dfinfo(dfbase, 'base')
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
dp.dfinfo(dfnew, 'col selection')
dfnew = dp.cleanup(dfnew, 'Test type', 'exclude_nan')
dp.dfinfo(dfnew, 'clean test type')

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


dfnew = dp.row_filter(dfnew, 'Laminate', ['MD2'], 'include')
dp.dfinfo(dfnew, 'filter on laminate')
dfnew = dp.row_filter(dfnew, 'Cut angle ', [0, 0.0, '0', '0.0'], 'include')
dp.dfinfo(dfnew, 'filter on 0 cut angle')

if 'Ncycles' in dfnew.columns:
    dfnew = dp.cleanup(dfnew, 'Ncycles', 'exclude_nan')
    dfnew = dp.row_filter(dfnew, 'Ncycles', [0.0, 0], 'exclude')
    dp.dfinfo(dfnew, 'clean ncycles')
if 'runout' in dfnew.columns:
    dfnew = dp.row_filter(dfnew, 'runout', ['y', 'Y'], 'exclude')
    dp.dfinfo(dfnew, 'remove runout rows')


# E column
if 'Eit' in dfnew.columns and 'Eic' in dfnew.columns:
    for i in dfnew.index:
        try:
            a = float(dfnew.loc[i, 'Eit'])
        except:
            a = np.nan
        try:
            b = float(dfnew.loc[i, 'Eic'])
        except:
            b = np.nan
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
    dp.dfinfo(dfnew, 'after E processing')

if stat_ref:
    stat_comp_ref = dp.row_filter(dfnew, 'Test type', ['STC'], 'include')
    stat_tens_ref = dp.row_filter(dfnew, 'Test type', ['STT'], 'include')
dfnew = dp.row_filter(dfnew, 'Test type', ['CA'], 'include')
dp.dfinfo(dfnew, 'filter on test type')
# static+ fatigue test processing
if statandfatigue:
    for i in dfnew.index:
        if dfnew.loc[i, 'Test type'] == 'STT' or dfnew.loc[i, 'Test type'] == 'STC':
            dfnew.loc[i, 'R-value1'] = 1.0
            dfnew.loc[i, 'f'] = 0.0
# end of stat+fatigue

collums_to_exclude = ['Test type', 'Laminate', 'Eic', 'Environment', 'Lab', 'runout']
dfnew = dp.big_cleanup(dfnew, 'exclude', collums_to_exclude)
if stat_ref:
    extra_excl = ['runout', 'f', 'R-value1', 'Temp.', 'Ncycles']
    stat_comp_ref = dp.big_cleanup(stat_comp_ref, 'exclude_nan', exclude_cols=collums_to_exclude + extra_excl)
    stat_tens_ref = dp.big_cleanup(stat_tens_ref, 'exclude_nan', exclude_cols=collums_to_exclude + extra_excl)
    col_to_compare = ['taverage', 'waverage', 'Lnominal', 'Fibre Volume Fraction', 'Cut angle ', 'Laminate']
    tens_str = dp.find_similar(dfnew, stat_tens_ref, col_to_compare, col_to_return=['Fmax', 'smax'], max_error_percent=5)
    comp_str = dp.find_similar(dfnew, stat_comp_ref, col_to_compare, col_to_return=['Fmax', 'smax'], max_error_percent=5)
    dfnew['stens'] = tens_str['smax']
    dfnew['ftens'] = tens_str['Fmax']
    dfnew['scomp'] = comp_str['smax']
    dfnew['fcomp'] = comp_str['Fmax']
    dp.dfinfo(dfnew, 'after stat ref')

dfnew = dp.col_filter(dfnew, collums_to_exclude, 'exclude')
dp.dfinfo(dfnew, 'exclude unneeded ,cols')
# directly related cols
if no_related_cols:
    rel_cols = ['smax', 'area', 'stens', 'scomp']
    dfnew = dp.col_filter(dfnew, rel_cols, 'exclude')
    dp.dfinfo(dfnew, 'removed directly related cols')

dfnew = dp.big_cleanup(dfnew, 'exclude', collums_to_exclude)
# log of some columns
if 'Ncycles' in dfnew.columns:
    dfnew['Ncycles'] = np.log10(dfnew['Ncycles'])
# random changes
if no_constant_cols:
    dfnew = dp.remove_constant_cols(dfnew)
print()
dp.dfinfo(dfnew, 'Final stats')
print(dfnew.dtypes)


name = 'data'
if tag == '':
    tag = dp.timetag()
name = name + tag + '.csv'

print()
if saveresult:
    dfnew.to_csv('processed\\' + name)
    print('File saved as: ' + name)
else:
    print('results not saved')