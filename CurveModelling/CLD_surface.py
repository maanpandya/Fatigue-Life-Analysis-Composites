import scipy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm

import CLD_definition

"""processing - x and y chosen because of similiar scales, otherwise it doesn't work
x  - mean stress
y  - stress amplitude
z  - log number of cycles

plotting - to match the video
x-axis - mean stress
y-axis - log number of cycles
z-axis - stress amplitude
"""

def makeSurface(dataframe):
    R_values, _, SN_models, _ = CLD_definition.CLD_definition(dataframe, plot=False)

    lives = [x/10. for x in range(1,80)]

    x = []
    y = []
    z = []

    for life in lives:
        for i in range(len(SN_models)):
            amp = 10**(float(SN_models[i].predict(np.array(life).reshape(-1, 1))))
            mean = CLD_definition.convert_to_mean_stress(amp,R_values[i])

            x.append(mean)
            y.append(amp)
            z.append(life)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    surface = sc.interpolate.Rbf(x,y,z)

    return surface

#Create a dataframe out of the csv file
dataframe = pd.read_csv("CurveModelling/Data/data42alt.csv")
CLD_definition.add_amplitudecol(dataframe)

R_values, R_slopes_coeff, SN_models, ax = CLD_definition.CLD_definition(dataframe, plot=False)

lives = [x/10. for x in range(1,80)]

x = []
y = []
z = []

for life in lives:
    for i in range(len(SN_models)):
        amp = 10**(float(SN_models[i].predict(np.array(life).reshape(-1, 1))))
        mean = CLD_definition.convert_to_mean_stress(amp,R_values[i])

        x.append(mean)
        y.append(amp)
        z.append(life)

x = np.array(x)
y = np.array(y)
z = np.array(z)

surface = makeSurface(dataframe)

xPlot, yPlot = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
zPlot = surface(xPlot, yPlot)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xPlot, zPlot, yPlot, cmap='viridis', alpha=0.5)
for i in range(len(SN_models)):
    ax.plot(x[i::len(R_values)], z[i::len(R_values)], y[i::len(R_values)])

ax.set_xlabel('Mean stress MPa')
ax.set_ylabel('log Number of cycles')
ax.set_zlabel('Stress amplitude MPa')

plt.show()


# Alternate RBFInterpolator code, doesn't plot

        # coords.append([mean, amp])

# coords = np.array(coords)

# surface = sp.interpolate.RBFInterpolator(coords, values)

# X = np.arange(-800, 800, 10)
# Y = np.arange(0, 800, 10)
# gridList= np.vstack(list(map(np.ravel, np.meshgrid(X, Y)))).T
# Z = surface(gridList)
# X, Y = np.meshgrid(X, Y)
# print(X,Y,Z)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(X, Y, Z)

# plt.show()s