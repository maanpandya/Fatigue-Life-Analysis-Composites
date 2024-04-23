import pandas as pd
import numpy as np

def separateDataFrameOLD(dataFrame, separationList = ["R-value1"]):
    """
    INPUT
        separationList - list of parameters(as strings) to separate by, "R-value1" by default
            
    """

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
            key = str(parameter) + " " + str(parameterValue) # create a key 
            parameterDictionary[key] = pd.DataFrame() # create empty dataframe
            for index, row in enumerate(dataFrame[parameter]):  # go thorugh the paramter column in the main dataframe
                if row == parameterValue: # if you encounter the desired value
                    parameterDictionary[key] = parameterDictionary[key]._append(dataFrame.iloc[index]) #add the row with the desired value to the output dataframe in the dictionary

    return parameterDictionary

def separateDataFrame(dataFrame, separationParameters = ["R-value1"], separationRanges = [False]):
    """
    INPUT \\
    dataFrame - dataframe on which the operation will be done \n
    separationParameters - list of parameters(as strings) to separate by, "R-value1" by default \n
        note: data2.csv has: nr,Fibre Volume Fraction,Cut angle ,taverage,waverage,area,Lnominal,R-value1,Ffatigue,smax,Ncycles,f,E,Temp. \n
    separationRanges - corresponding list of numbers seperating the ranges and/or the booleans False when ranges shoudnt be made \n
        example: separationParameters = ["R-value1","Temp."], separationRanges = [False, [0,10,20]] \n

    OUTPUT  \n
    parameterDictionary - dictionary with  \n
        keys - parameter by which the dataframe was separated  \n
        values - dictionary with values and separated data  \n
            keys - value or top of separation range  \n
            values - dataframe with test with the parameter of a certain value or within certain range  \n
    """
    
    parameterDictionary = dict() #create empty dictionary
    for index, parameter in enumerate(separationParameters):
        parameterDictionary[parameter] = dict() # create empty dictionary within dictionary
        if separationRanges[index] == False: # if there are no ranges
            groupBy = dataFrame.groupby(parameter) # create groupby object
            parameterDictionary[parameter] = dict(zip(list(groupBy.groups.keys()),[groupBy.get_group(x) for x in groupBy.groups])) 
        else:
            # groupBy = dataFrame.groupby(pd.cut(dataFrame[parameter], separationRanges[index])) Other method, broken
            # parameterDictionary[parameter] = dict(zip(["a","b","c"],[groupBy.get_group(x) for x in groupBy.groups])) 
            separationRanges[index].sort() # sort list in case it's not ascending
            for separationIndex in range(len(separationRanges[index])): # go through each bound
                parameterDictionary[parameter][separationRanges[index][separationIndex]] = pd.DataFrame() # create empty dataframe for the range with key of end of range
                for valueIndex, value in enumerate(dataFrame[parameter]): # go through the dataframe column for the parameter
                    print(separationRanges[index],separationIndex,separationRanges[index][separationIndex])
                    if value < separationRanges[index][separationIndex]: # if the value is smaller than the bound
                        if separationIndex == 0:
                            # print(value, "smaller than", separationRanges[index][separationIndex]) # Debugging
                            parameterDictionary[parameter][separationRanges[index][separationIndex]] = parameterDictionary[parameter][separationRanges[index][separationIndex]]._append(dataFrame.iloc[valueIndex])
                        elif value >= separationRanges[index][separationIndex - 1]: # and larger than the previous bound if such exists
                            # print(value, "in", separationRanges[index][separationIndex - 1], separationRanges[index][separationIndex]) # Debugging
                            parameterDictionary[parameter][separationRanges[index][separationIndex]] = parameterDictionary[parameter][separationRanges[index][separationIndex]]._append(dataFrame.iloc[valueIndex])                       
    return parameterDictionary

# TESTING
# dataFrame = pd.read_csv("CurveModelling\Data\data2.csv")
# WITH RANGES
# parameterDictionary = separateDataFrame(dataFrame, separationParameters = ["Temp."], separationRanges=[[-20.,10.,20.,40.]])
# print(parameterDictionary["Temp."])
# print("hi")

# WITHOUT RANGES
# parameterDictionary = separateDataFrame(dataFrame)
# print(parameterDictionary)