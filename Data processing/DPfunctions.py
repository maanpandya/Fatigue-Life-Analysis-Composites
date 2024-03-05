import pandas as pd
import numpy as np
import time

def cleanup(dataframe, collum, mode):
    n = 0
    Tsum = 0
    nanlist = []
    changelist = []
    for i in dataframe.index:
        i = int(i)
        j = dataframe[collum][i]
        if pd.isna(j) == False:
            try:
                j = float(j)
            except:
                changelist.append(i)
            else:
                n += 1
                Tsum += j
                dataframe.loc[i, collum] = j
        else:
            nanlist.append(i)
    if n > 0:
        avg = Tsum / n
    else:
        avg = np.nan
    for i in nanlist:
        if mode == 'avg' or mode == 'avg_manual':
            dataframe.loc[i, collum] = avg
        if mode == 'exclude' or mode == 'exclude_manual' or mode == 'exclude_nan':
            dataframe = pd.DataFrame.drop(dataframe, i)
            
    if mode != 'exclude_nan':
        print('amount of sinkable = ' + str(len(changelist)))

    for i in changelist:
        if mode == 'avg':
            dataframe.loc[i, collum] = avg
        elif mode == 'avg_manual' or mode == 'exclude_manual':
            j = input('current value: ' + str(dataframe.loc[i, collum]) + ', new value (leave empty to excl row) = ')
            if j == '':
                dataframe = pd.DataFrame.drop(dataframe, i)
            else:
                dataframe.loc[i, collum] = float(j)
        elif mode == 'exclude':
            dataframe = pd.DataFrame.drop(dataframe, i)
    return dataframe

def col_filter(dataframe, columns, mode):
    dfnew = pd.DataFrame(dataframe.index)
    dfnew = dfnew.set_index(dfnew.columns[0])
    for i in dataframe.columns:
        if i in columns and mode == 'include':
            dfnew = pd.DataFrame.join(self=dfnew, other=dataframe[i])
        elif i not in columns and mode == 'exclude':
            dfnew = pd.DataFrame.join(self=dfnew, other=dataframe[i])
    return dfnew

def dfinfo(dataframe):
    print(str(len(dataframe.index)) + ' rows x ' + str(len(dataframe.columns)) + ' columns ')
    return [len(dataframe.index), len(dataframe.columns)]
