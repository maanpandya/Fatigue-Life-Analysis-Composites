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
    'Ffatigue', 'Ncycles', 'smax'
]
dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Test type', 'exclude_nan')
dp.dfinfo(dfnew)
dfnew = dp.row_filter(dfnew, 'Test type', 'CA', 'include')
dfnew = dp.col_filter(dfnew,['Test type'], 'exclude')
dp.dfinfo(dfnew)



#pd.DataFrame.to_csv(path_or_buf='smaxdata.csv', self=dataframe['smax'], index=False)