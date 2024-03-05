import pandas as pd
import numpy as np
import time
import DPfunctions as dp


file = 'optimatforpy.csv'
dfbase = pd.read_csv(file)
dfbase = dfbase.set_index('nr')

print(dfbase.head(10))
dp.dfinfo(dfbase)

collums_to_include = [
    'taverage', 'waverage', 'Lnominal','Test type',
    'Temp.', 'Fibre Volume Fraction', 'R-value1',
    'Ffatigue', 'Ncycles', 'smax', 'f', 'Laminate',
    'Eit', 'Eic'
]
dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Test type', 'exclude_nan')
dfnew = dp.row_filter(dfnew, 'Test type', ['CA'], 'include')
dfnew = dp.cleanup(dfnew, 'Ncycles', 'exclude_nan')
dfnew = dp.row_filter(dfnew, 'Laminate', ['UD1', 'UD2', 'UD3', 'UD4'], 'include')

dfnew = dp.cleanup(dfnew, 'Temp.', 'avg')
dfnew = dp.row_filter(dfnew, 'Fibre Volume Fraction', ['#N/A'], 'exclude')

newc = np.zeros(len(dfnew.index))

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

dfnew = dp.col_filter(dfnew,['Test type', 'Laminate', 'Eic'], 'exclude')
dfnew = dfnew.rename(columns={'Eit': 'E'})
dfnew = dp.cleanup(dfnew, 'E', 'avg')


for i in dfnew.columns:
    b = len(dfnew.index)
    dfnew = dp.cleanup(dfnew, i, 'exclude')
    if len(dfnew.index) / b < 0.95:
        print(i + ' had a lot of nan')
    dfnew[i] = dfnew[i].astype(dtype=float)

print(dfnew.dtypes)
dp.dfinfo(dfnew)

name = 'processed\\testdata'+str(time.time()) + '.csv'
pd.DataFrame.to_csv(dfnew, name)





#pd.DataFrame.to_csv(path_or_buf='smaxdata.csv', self=dataframe['smax'], index=False)