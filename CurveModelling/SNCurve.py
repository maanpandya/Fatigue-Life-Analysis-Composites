import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def regression(nArray, sArray):
    """
    Conduct linear regression, taking an array of fatigue life in log10 of N and Stress in MPa \n
    INPUT \n
        nArray - array of log10 of fatigue life(number of cycles) \n
        sArray - array of stress in MPa, not log10 \n
    This is so that you can use the columns of the dataframe since they are formatted as shown above\n
    OUTPUT \n
        model - model that relates number of cycles in log10(input) to stress in log10(output)
    """
     
    from sklearn.linear_model import LinearRegression

    nArray = nArray.reshape((-1,1))
    sArray = np.log10(np.absolute(sArray))
    model = LinearRegression().fit(nArray, sArray)
    # print(model.get_params()) # debugging

    return model

# Plotting S-N Curve with prediction error
def plotStd(X,Y, dev=3):
    """
    Under construction, not sure if even needed as a function; has useful stuff inside though, works as a nice example \n
    Plot linear regression curve with uncertainties \n
    INPUT \n
        X - x values, for S-N curve, log10 of fatigue life, column is already log10 in data \n
        Y - y values, for S-N curve, log10 of stress, need to take log10 of column \n
        std - number of standard deviations
    """
    X_with_intercept = sm.add_constant(X)

    model = sm.OLS(Y, X_with_intercept).fit()

    predictions = model.predict(X_with_intercept)
    Residuals = Y-predictions

    std_dev = np.std(Residuals)*dev

    plt.scatter(X,Y, color = 'lightblue')

    params = model.params
    plt.plot(X, params[0] + params[1]*X, color='blue', label='Regression Line')

    plt.plot(X, (params[0] + params[1]*X) + std_dev, color='orange', label='STD')
    plt.plot(X, (params[0] + params[1]*X) - std_dev, color='orange')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression with Statsmodels')
    plt.legend()

    ax = plt.gca()# get the current axes

    return ax

# # DEBUGGING
# X = np.array([1,4,6,7,10,17]).reshape((-1, 1))
# Y = np.array([4, 6, 8, 10, 5, 12])
# plotStd(X,Y)


# # DEBUGGING
# dfMain = pd.read_csv("CurveModelling/Data/data2.csv")
# from Data_processing import separateDataFrame
# dictionary = separateDataFrame(dfMain, separationParameters=["R-value1","Cut angle "], separationRanges=[False,False])
# print(dictionary["Cut angle "].keys())

# # df1 = pd.merge(dictionary["R-value1"][-1],dictionary["Cut angle "][0.])
# df1 = dictionary["R-value1"][-1]
# model = regression(np.array(df1["Ncycles"]), np.array(df1["smax"]))
# x1 = np.linspace(0,10)
# x2 = model.predict(x1.reshape(-1,1))
# plt.plot(x1, x2)
# df1 = pd.merge(dictionary["R-value1"][-1],dictionary["Cut angle "][0.])
# model = regression(np.array(df1["Ncycles"]), np.array(df1["smax"]))
# x1 = np.linspace(0,10)
# x2 = model.predict(x1.reshape(-1,1))
# plt.plot(x1, x2)

# plt.scatter(pd.merge(dictionary["R-value1"][-1], dictionary["Cut angle "][0.0])["Ncycles"], np.log10(pd.merge(dictionary["R-value1"][-1],dictionary["Cut angle "][0.0])["smax"]), c="green")
# plt.scatter(pd.merge(dictionary["R-value1"][-1], dictionary["Cut angle "][10.0])["Ncycles"], np.log10(pd.merge(dictionary["R-value1"][-1],dictionary["Cut angle "][10.0])["smax"]), c="yellow")
# plt.scatter(pd.merge(dictionary["R-value1"][-1], dictionary["Cut angle "][60.0])["Ncycles"], np.log10(pd.merge(dictionary["R-value1"][-1],dictionary["Cut angle "][60.0])["smax"]), c="purple")
# plt.scatter(pd.merge(dictionary["R-value1"][-1],dictionary["Cut angle "][90.0])["Ncycles"], np.log10(pd.merge(dictionary["R-value1"][-1],dictionary["Cut angle "][90.0])["smax"]), c="blue")
# plt.show()

