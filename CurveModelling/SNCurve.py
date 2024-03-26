import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def regression(nArray, sArray):
    #  Conduct linear regression, taking an array of fatigue life in log10 of N and Stress in MPa
    from sklearn.linear_model import LinearRegression

    nArray = nArray.reshape((-1,1))
    sArray = np.log10(sArray)
    model = LinearRegression().fit(nArray, sArray)

    return model
 
dfMain = pd.read_csv("CurveModelling/Data/data2.csv")
from Data_processing import separateDataFrame
dictionary = separateDataFrame(dfMain, separationList=["R-value1", "Fibre Volume Fraction"])
R1N = np.array(dictionary["R-value1 -1.0"]["Ncycles"])
R1S = np.array(dictionary["R-value1 -1.0"]["smax"])
print(dictionary)

model = regression(R1N, R1S)
x1 = np.linspace(0,10)
x2 = model.predict(x1.reshape(-1,1))

plt.plot(x1, x2)
plt.scatter(R1N, np.log10(R1S), c="green")
plt.scatter(dictionary["Fibre Volume Fraction 0.0"]["Ncycles"], np.log10(dictionary["Fibre Volume Fraction 0.0"]["smax"]), c="lightblue")
plt.show()

