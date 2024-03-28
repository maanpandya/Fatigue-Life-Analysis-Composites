import pandas as pd

def separateDataFrameOLD(dataFrame, separationList = ["R-value1"]):

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
            key = str(parameter) + " " + str(parameterValue) # create a key 
            parameterDictionary[key] = pd.DataFrame() # create empty dataframe
            for index, row in enumerate(dataFrame[parameter]):  # go thorugh the paramter column in the main dataframe
                if row == parameterValue: # if you encounter the desired value
                    parameterDictionary[key] = parameterDictionary[key]._append(dataFrame.iloc[index]) #add the row with the desired value to the output dataframe in the dictionary

    return parameterDictionary

def separateDataFrame(dataFrame, separationList = ["R-value1"]):
    groupBy = dataFrame.groupby(separationList)
    return [groupBy.get_group(x) for x in groupBy.groups]


# dataFrame = pd.read_csv("CurveModelling\Data\data2.csv")
# parameterDictionary = separateDataFrame(dataFrame, separationList = ["R-value1", "Temp."])
# print(parameterDictionary)