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
            
    if mode != 'exclude_nan' and len(changelist) > 0:
        print('amount of sinkable in ' + collum + '= ' + str(len(changelist)))

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

def row_filter(dataframe, column, filters, mode):
    if mode == 'include':
        for i in dataframe.index:
            j = dataframe.loc[i, column]
            if j not in filters:
                dataframe = pd.DataFrame.drop(dataframe, i)
    if mode == 'exclude':
        for i in dataframe.index:
            j = dataframe.loc[i, column]
            if j in filters:
                dataframe = pd.DataFrame.drop(dataframe, i)
    return dataframe

def timetag(format=False, day=True, currtime=True, bracket=False):
    f = ''
    if format:
        if day:
            f = f + '%d-%m-%y '
        if currtime:
            f = f + '%H:%M:%S'
    else:
        if day:
            f += '%d%m%y'
        if currtime:
            f += '%H%M%S'
    f = f.strip()
    if bracket:
        f = '('+f+')'
    return time.strftime(f, time.localtime())

def datasplitscale(dataframe, test_size=0 , exclude_columns=[]):
    from sklearn.model_selection import train_test_split
    dftrain, dftest = train_test_split(dataframe, test_size=test_size)
    scalers = {}
    for i in dftrain.columns:
        if i not in exclude_columns:
            mean = np.mean(dftrain[i])
            std = np.std(dftrain[i])
            dftrain[i] = (dftrain[i] - mean) / std
            dftest[i] = (dftest[i] - mean) / std
        else:
            mean = 0
            std = 1
        scalers[i] = {'mean': mean, 'std': std}
    return dftrain, dftest, scalers

def dfread(file):
    base = pd.read_csv(file)
    base = base.set_index(base.columns[0])
    return base

def dfxysplit(dataframe, y_columns=[]):
    '''x = col_filter(dataframe, y_columns, 'exclude')
    y = col_filter(dataframe, y_columns, 'include')'''

    x = dataframe.drop(columns=y_columns)
    y = dataframe[y_columns]
    return x, y