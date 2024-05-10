import pandas as pd
import numpy as np
import time
import sympy as sp
from warnings import warn


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

def dfinfo(dataframe, tag=''):
    print(tag + ' : ' + str(len(dataframe.index)) + ' rows x ' + str(len(dataframe.columns)) + ' columns ')
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


def row_filter_remove(dataframe, column, filters, mode):
    df = row_filter(dataframe, column, filters, mode)
    return col_filter(df, [column], 'exclude')

def label_col(dataframe, column, possible_values=None):
    if possible_values == None:
        possible_values = []
        for i in dataframe.index:
            val = dataframe[column].loc[i]
            if val not in possible_values:
                possible_values.append(val)
    labels = {}
    for i in range(len(possible_values)):
        labels[possible_values[i]] = float(i)
    col_out = []
    for i in dataframe.index:
        col_out.append(labels[dataframe[column].loc[i]])
    dataframe[column] = col_out
    return dataframe, labels


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
            if std == 0.0 or std == 0:
                std = 1
                print('std of col', i, 'is 0, with mean', mean, 'whole column will be zero')
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

def big_cleanup(dataframe, mode='exclude', exclude_cols=[]):
    print('big cleanup')
    for i in dataframe.columns:
        if i not in exclude_cols:
            b = len(dataframe.index)
            dataframe = cleanup(dataframe, i, mode)
            if len(dataframe.index) / b < 1:
                print(i + ': ' + str(b - len(dataframe.index)) + '/' + str(b) + ' removed')
            dataframe[i] = dataframe[i].astype(dtype=float)
    return dataframe


def find_similar(df1, df2, col_to_compare, col_to_return=[], max_error_percent=1):
    # find similar rows in df2 for every row in df1, only looking at col_to_compare, and returning a dataframe with
    # same index as df1 with data from col to return, if col_to_return is empty, return list of similar indexes
    returnindex = False
    if col_to_return == []:
        col_to_return.append('indexlists')
        returnindex = True
    df_return = pd.DataFrame(columns=col_to_return, index=df1.index)
    err = max_error_percent / 100
    simlstsizes = []
    for i in df1.index:
        row1 = np.array(df1.loc[i].loc[col_to_compare])
        sim_lst = []
        err_lst = []
        for j in df2.index:
            is_sim = True
            totcurerr = 0
            row2 = np.array(df2.loc[j].loc[col_to_compare])
            for k in range(len(row1)):
                #print(k, ' - ', type(row1[k]))
                val1 = row1[k]
                val2 = row2[k]
                if type(row1[k]) == np.float64:
                    if row1[k] == 0:
                        if row1[k] != row2[k]:
                            is_sim = False
                            break
                    else:
                        curr_err = np.abs((row2[k] - row1[k]) / row1[k])
                        if curr_err > err:
                            is_sim = False
                            break
                        else:
                            totcurerr += curr_err
                else:
                    if row1[k] != row2[k]:
                        is_sim = False
                        break
            if is_sim:
                sim_lst.append(j)
                err_lst.append(totcurerr)

        if returnindex and sim_lst !=[]:
            df_return.loc[i, 'indexlists'] = sim_lst
        elif len(sim_lst) >= 1:
            simlstsizes.append(len(sim_lst))
            m = np.argmin(np.array(err_lst))
            for l in col_to_return:
                df_return.loc[i, l] = df2.loc[sim_lst[m], l]
    if not returnindex:
        print('similar result:' + str(np.mean(np.array(simlstsizes))))
        dfinfo(df_return.dropna())
    return df_return

def remove_constant_cols(dataframe):
    remove = []
    for i in dataframe.columns:
        std = np.std(dataframe[i])
        mean = np.mean(dataframe[i])
        if std == 0:
            print('col', i, 'has constant value of', mean, 'and will be removed')
            remove.append(i)
    dataframe = col_filter(dataframe, remove, 'exclude')
    return dataframe


def filter_dataframe_by_cutoff(df, column, cutoff):
    # Filter rows above the cutoff
    above_cutoff = df[df[column] >= cutoff]

    # Filter rows below or equal to the cutoff
    below_cutoff = df[df[column] < cutoff]

    return above_cutoff, below_cutoff


def rmath(inputs: dict, output: str):
    # inputs: dict => {'R':2, 'smax':400}
    # output:str => 'smean'
    # using sensible smax inputs (smax is always maximum stress and smax is always larger than smin)
    if len(inputs) != 2:
        raise Exception(f'can only compute with 2 knowns, got {len(inputs)}.')
    known = []
    unknown = [output]
    for i in inputs:
        known.append(i)
    smin, smax, R, samp, smean = sp.symbols('smin, smax, R, samp, smean', real=True)
    symdict = {'smin':smin, 'smax':smax, 'R':R, 'samp':samp, 'smean':smean}
    eq1 = smin/smax - R
    eq2 = smax+smin - 2*smean
    eq3 = smax-smin - 2*samp
    symlist = [symdict[output]]
    for i in symdict:
        if i not in known and i is not output:
            symlist.append(symdict[i])
            unknown.append(i)
    sols = sp.nonlinsolve([eq1, eq2, eq3], symlist)
    sols = list(sols)[0]
    f = sp.utilities.lambdify([symdict[known[0]], symdict[known[1]]], sols[0])
    try:
        out = f(inputs[known[0]], inputs[known[1]])
    except ZeroDivisionError:
        # math breaking error handling
        warn(f'division by zero trying to calculate {output}, assumed output is 0.')
        out = 0
    if type(out) is not float or int:
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out