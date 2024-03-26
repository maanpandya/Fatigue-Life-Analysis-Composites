import pandas as pd
import numpy as np

# Read csv
dfMain = pd.read_csv("Data/data2.csv")

def separateDataFrame(dataFrame, separationList = ["R-value1"]):

    # separationList - Parameters to separate by, r value by default
    # data2.csv has nr,Fibre Volume Fraction,Cut angle ,taverage,waverage,area,Lnominal,R-value1,Ffatigue,smax,Ncycles,f,E,Temp.

    # set up dictionary with keys ("parameter" + " " + "parameterValue" ) and the list of values for that parameter and marameterValue
    parameterDictionary = dict()

    for parameter in separationList:  # for each parameter that needs to be separated

        # Set up a list of the possible parameter values
        parameterValues = [dataFrame[parameter][0]]  # get one starting parameter value
        for index, row in enumerate(dataFrame[parameter]):
            counter = 0
            for parameterValue in parameterValues:
                if row != parameterValue:
                    counter += 1
                    if counter == len(parameterValues):  # if you find that there are more parameters that don't
                        # match than you know of
                        parameterValues.append(row)

        for parameterValue in parameterValues:  # for each parameter value
            key = str(parameter) + " " + str(parameterValue)
            parameterDictionary[key] = pd.DataFrame()
            for index, row in enumerate(dataFrame[parameter]):
                if row == parameterValue:
                    parameterDictionary[key] = parameterDictionary[key]._append(dataFrame.iloc[index])

        return parameterDictionary

dictionary = separateDataFrame(dfMain)
print(dictionary)
