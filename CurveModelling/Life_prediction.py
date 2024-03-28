import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Data_processing import separateDataFrame
#from SNCurve import SN_models

"""
GOAL:
Find the fatigue life of a material for any given load ratio and amplitude

METHOD:
Create areas of the CLD curve for the available load ratios by using the SN regression models
Create a linear interpolation model for each of the areas for a given N falue
Interpolate the target inside the area

IMPUTS:
SN regression models for all the R values.
Target stress amplitude and mean stress

Output:
Fatigue life for the given stress amplitude and mean stress
"""

#### IMPUT

target_stress_amplitude = 100
target_mean_stress = 50

#####


def Load_R_line_creator(model,N_to_connect):
    intersect = 0
    slope = model(N_to_connect)/N_to_connect

    return slope,intersect

def R_line_visualizer(R_slopes_coeff,R_values):
    fig, ax = plt.subplots()

    for slope in R_slopes_coeff:
        if slope[0] >= 0:
            x = np.linspace(0,600,2)

        else:
            x = np.linspace(-600,0,2)

        y = slope[0]*x + slope[1]

        plt.plot(x, y, label=f"R = {R_values[i]}")

    plt.show()
    return


#------------------- Split the database into dictionaries by R values and store which R values are available
dataframe = pd.read_csv("data2.csv")
separateDataFrame(dataframe, separationList = ["R-value1"])


#------------------- Create the slopes and intersects for each R value to define the R lines


#Define intersection point for the slopes
N_to_connect = 10**3

#Create a list of slopes and intersects for each R value
R_slopes_coeff = []

#Add slope 0 
R_slopes_coeff.extend([0,0])

for R in range(len(SN_models)):
    slope, intersect = Load_R_line_creator(SN_models[i])
    R_slopes_coeff.extend([slope,intersect])

#Add slope 0 
R_slopes_coeff.extend([0,0])

#Reshape to a 2D array
R_slopes_coeff = np.array(R_slopes_coeff).reshape(-1,2)

#------------------- Find where the target is in the CLD curve

if target_stress_amplitude < 

for slope in R_slopes_coeff:
    area = 
    if 

    if slope[0]*target_mean_stress + slope[1] > target_stress_amplitude:
        break
Slope_values = 
