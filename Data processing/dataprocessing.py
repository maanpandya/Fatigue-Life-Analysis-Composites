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
    'Ffatigue', 'Ncycles', 'smax', 'f', 'Laminate'
]
dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Test type', 'exclude_nan')
dp.dfinfo(dfnew)
dfnew = dp.row_filter(dfnew, 'Test type', ['CA'], 'include')

dfnew = dp.cleanup(dfnew, 'Ncycles', 'exclude_nan')
dfnew = dp.row_filter(dfnew, 'Laminate', ['UD1', 'UD2', 'UD3', 'UD4'], 'include')
dfnew = dp.col_filter(dfnew,['Test type', 'Laminate'], 'exclude')

print(dfnew.columns)
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Temp.', 'avg')
dfnew = dp.row_filter(dfnew, 'Fibre Volume Fraction', ['#N/A'], 'exclude')


for i in dfnew.columns:
    b = len(dfnew.index)
    dfnew = dp.cleanup(dfnew, i, 'exclude')
    if len(dfnew.index) / b < 0.95:
        print(i + 'had a lot of nan')
    for j in dfnew.index:
        try:
            k = float(dfnew.loc[j, i])
        except:
            pd.DataFrame.drop(dfnew, j)
        else:
            dfnew.loc[j, i] = float(dfnew.loc[j, i])
    dfnew[i] = dfnew[i].astype(dtype=float)
name = 'processed\\testdata'+str(time.time()) + '.csv'
pd.DataFrame.to_csv(dfnew, name)



print(dfnew.dtypes)
dp.dfinfo(dfnew)


#pd.DataFrame.to_csv(path_or_buf='smaxdata.csv', self=dataframe['smax'], index=False)