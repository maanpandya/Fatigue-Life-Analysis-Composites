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
dfnew = dp.col_filter(dfnew,['Test type'], 'exclude')
dfnew = dp.cleanup(dfnew, 'Ncycles', 'exclude_nan')
#dfnew = dp.row_filter(dfnew, 'Laminate', ['UD1', 'UD2', 'UD3', 'UD4'], 'include')
dp.dfinfo(dfnew)
print(dfnew.columns)



#pd.DataFrame.to_csv(path_or_buf='smaxdata.csv', self=dataframe['smax'], index=False)