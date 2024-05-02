import scipy as sp
import numpy as np
import pandas as pd

import CLD_definition

#Create a dataframe out of the csv file
dataframe = pd.read_csv("CurveModelling/Data/data42alt.csv")
UTS = 820
UCS = -490
CLD_definition.add_amplitudecol(dataframe)

R_values, R_slopes_coeff, SN_models, ax = CLD_definition.CLD_definition(dataframe, plot=False)

lives = [x/10. for x in range(1,100)]

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

xPlot, yPlot = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
zPlot = surface(xPlot, yPlot)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xPlot, zPlot, yPlot, cmap='viridis', alpha=0.5)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf']
for i in range(len(SN_models)):
    ax.plot(x[i::7], z[i::7], y[i::7], c=colors[i])

ax.set_xlabel('mean stress')
ax.set_ylabel('log number of cycles')
ax.set_zlabel('stress amplitude')

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