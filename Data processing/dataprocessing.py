import pandas as pd
import numpy as np
import time
import DPfunctions as dp


file = 'optimatforpy.csv'
dfbase = pd.read_csv(file)
dfbase = dfbase.set_index('nr')

print(dfbase.head(10))
dp.dfinfo(dfbase)

collums_to_include = ['taverage', 'waverage', 'Lnominal','Test type']
dfnew = dp.col_filter(dfbase, collums_to_include, 'include')
print(dfnew.head())
dp.dfinfo(dfnew)
dfnew = dp.cleanup(dfnew, 'Test type', 'exclude_nan')
print(dfnew.head())
dp.dfinfo(dfnew)

#dataframe = nan_avg(dataframe, 'smax', 'manual')

#pd.DataFrame.to_csv(path_or_buf='smaxdata.csv', self=dataframe['smax'], index=False)