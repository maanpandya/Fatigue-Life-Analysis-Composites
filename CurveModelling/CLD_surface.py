import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm

import CLD_definition

# x axis - mean stress
# y axis - stress amplitude
# value - number of cycles


#Create a dataframe out of the csv file
dataframe = pd.read_csv("CurveModelling/Data/altdata.csv")
UTS = 820
UCS = -490
CLD_definition.add_amplitudecol(dataframe)

R_values, R_slopes_coeff, SN_models, ax = CLD_definition.CLD_definition(dataframe, plot=True)

lives = [x/10. for x in range(1,10)]
coords = []
values = []


for life in lives:
    for i in range(len(SN_models)):
        amp = 10**(float(SN_models[i].predict(np.array(life).reshape(-1, 1))))
        mean = CLD_definition.convert_to_mean_stress(amp,R_values[i])
        coords.append([mean, amp])
        values.append(life)

coords = np.array(coords)
values = np.array(values)
print(coords, values)

surface = sp.interpolate.RBFInterpolator(coords, values)

X = np.arange(-800, 800, 10)
Y = np.arange(0, 800, 10)
gridList= np.vstack(list(map(np.ravel, np.meshgrid(X, Y)))).T
Z = surface(gridList)
X, Y = np.meshgrid(X, Y)
print(X,Y,Z)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z)

plt.show()

