import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Data_processing import separateDataFrame
from SNCurve import regression
np.set_printoptions(precision=4)
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


def convert_to_mean_stress(amp,R):
    S_max = amp*2/(1-R)
    S_min = S_max*R
    Mean = (S_max + S_min)/2
    
    return Mean

def Slope_R_line_creator(model,R,N_to_connect):
    intersect = 0
    N_to_connect = np.array(N_to_connect).reshape(-1, 1)
    amp = 10**(float(model.predict(N_to_connect)))
    Mean = convert_to_mean_stress(amp,R)

    if Mean == 0:
        slope = 10**8
    
    else:
        slope = amp/Mean

    return slope,intersect

def R_line_visualizer(R_slopes_coeff,R_values,ax):

    i = 0
    for slope in R_slopes_coeff:
        if slope[0] >= 0:
            x = np.linspace(0,600,2)

        else:
            x = np.linspace(-600,0,2)

        y = slope[0]*x + slope[1]

        plt.plot(x, y, label=f"R = {R_values[i]}")
        i += 1

    ax.set_ylim(0, 300)
    return


#################### Data Processing (Jan tasks)

#Create a dataframe out of the csv file
dataframe = pd.read_csv("CurveModelling/Data/data2.csv")

#Find which R values are available
R_values = list(dataframe.groupby("R-value1").groups.keys())
print("R values available: ", R_values)

#Separate the dataframe by R values and range of temperature
parameterDictionary = separateDataFrame(dataframe, separationParameters= ["R-value1", "Temp."], separationRanges=[False, [0,30]]) # placeholder temperature

#Create a list of SN models for each R value and the selected temperature range
SN_models = []
for key, df in parameterDictionary["R-value1"].items(): # go through the dataframe for each r-value
    df = pd.merge(df, parameterDictionary["Temp."][30]) # merge/take overlap of each dataframe with the desired temperature (placeholder temperature)
    SN_models.append(regression(np.array(df["Ncycles"]), np.array(df["smax"])))

print("Number of regression models available: ", len(SN_models))

#------------------- Create the slopes and intersects for each R value to define the R lines

#Define intersection point for the slopes
N_to_connect = 10**3
print("\nSlopes will be created in tersection the number of cycles to failure: ", N_to_connect)

#Create a list of slopes and intersects for each R value
R_slopes_coeff = []

for m in range(len(SN_models)):
    slope, intersect = Slope_R_line_creator(SN_models[m],R_values[m],np.log10(N_to_connect))
    R_slopes_coeff.append([slope,intersect])

#Reshape to a 2D array
R_slopes_coeff = np.array(R_slopes_coeff)
print("The list of slopes and intersects for each R value:\n")
print(R_slopes_coeff)

#Creates a list to sort the R lines by the inverse of the slope
List_for_sorting = np.power(R_slopes_coeff[:, 0],-1)

#Find the indices that would sort the list
sort_indices = np.argsort(List_for_sorting)

#Sort the R_values , R_slopes_coeff and the SN_models
R_slopes_coeff = R_slopes_coeff[sort_indices]
SN_models = np.array(SN_models)[sort_indices]
R_values = np.array(R_values)[sort_indices]

#------------------- Find where the target is in the CLD curve
target_stress_amplitude = 250
target_mean_stress = -200

#Find the amplitude of the slopes for the given target mean stress
Slopes_amp = []
for slope in R_slopes_coeff:
    amp = slope[0]*target_mean_stress + slope[1]
    Slopes_amp.append(amp)

#Add the amplitude of the slope 0
Slopes_amp = np.array(Slopes_amp)

Dis_abs = np.abs(target_stress_amplitude - Slopes_amp)
Dis = target_stress_amplitude - Slopes_amp

min_index = np.argmin(Dis_abs)

if Dis[min_index] > 0:
    print(f"Target is above R = {R_values[min_index]}")

else:
    print(f"Target is below R = {R_values[min_index]}")

#------------------- Create constant life lines

Life_lines_log = [3,4,5,6]
amp_plot_lists = []
mean_plot_lists = []

for life in Life_lines_log:
    amp_list = []
    mean_list = []

    for i in range(len(SN_models)):
        amp = 10**(float(SN_models[i].predict(np.array(life).reshape(-1, 1))))
        mean = convert_to_mean_stress(amp,R_values[i])
        amp_list.append(amp)
        mean_list.append(mean)

    amp_plot_lists.append(amp_list)
    mean_plot_lists.append(mean_list)


#------------------ Visualize the CLD graph
fig, ax = plt.subplots()
ax.scatter(target_mean_stress,target_stress_amplitude, c="red", label="Target")

R_line_visualizer(R_slopes_coeff,R_values,ax)

for p in range(len(amp_plot_lists)):
    ax.plot(mean_plot_lists[p], amp_plot_lists[p], label=f"Life = {Life_lines_log[p]}")
    ax.legend()

plt.show()