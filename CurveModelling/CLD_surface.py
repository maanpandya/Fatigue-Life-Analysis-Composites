import scipy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def makeSurface(R_values,SN_models, lives = [x/10. for x in range(1,80)], dy = [] ):
    """Create Radial Basis Function interpolated surface from the data in the dataframe\n
    INPUT \n
    R-values \n
    SN_models \n
    OUTPUT \n
    surface - function: takes two arguments - mean stress in MPa and Stress Amplitude in MPa and returns log number of cycles \n
    x,y,z - sampled points of the S-N curves, for plotting
    """
    print("-----------------------------")
    print("Creating CLD surface...")
    print("\n")

    x = []
    y = []
    z = []


    for index in range(len(R_values)): # for each R value
        if len(dy) > 0:
            amp = 10**((SN_models[index].predict(np.array(lives).reshape(-1, 1))) + dy[index])
        else:
            amp = 10**((SN_models[index].predict(np.array(lives).reshape(-1, 1))))
                       
        mean = CLD_definition.convert_to_mean_stress(amp,R_values[index])

        x.append(mean)
        y.append(amp)
        z.append(lives)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    surface = sc.interpolate.Rbf(x,y,z)

    print("\nSurface created.")
    print("-----------------------------\n")

    return surface,x,y,z

def plotSurface(SN_models,R_values,surface,x,y,z):
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

    return ax