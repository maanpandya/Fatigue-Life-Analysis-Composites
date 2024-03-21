import pandas as pd
import numpy as np

# Read csv
dfmain = pd.read_csv("C:\Jan\Studia\Works\Year 2\Test, Analysis & Simulation\Code\Fatigue-Life-Analysis-Composites\CurveModelling\Data\data2.csv")

# Parameters to seperate by
# data2.csv has nr,Fibre Volume Fraction,Cut angle ,taverage,waverage,area,Lnominal,R-value1,Ffatigue,smax,Ncycles,f,E,Temp.
seperation = ["R-value1"]

parameterDictionary = dict()

for parameter in seperation:

    # Set up a list of the possible parameter values
    parameterValues = [dfmain[parameter][0]]
    for index, row in enumerate(dfmain[parameter]):
        counter = 0
        for parameterValue in parameterValues:
            if row != parameterValue:
                counter += 1
                if counter == len(parameterValues):
                    parameterValues.append(row)             

    for parameterValue in parameterValues:
        key = str(parameter) + " " + str(parameterValue)
        parameterDictionary[key] = pd.DataFrame()
        for index, row in enumerate(dfmain[parameter]):
            if row == parameterValue:
                parameterDictionary[key] = parameterDictionary[key]._append(dfmain.iloc[index])

