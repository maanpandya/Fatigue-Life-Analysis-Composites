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
Create a linear interpolation model for each of the areas for a given N value
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
    Y_limit = 600
    i = 0
    for slope in R_slopes_coeff:
        if R_values[i] != -1:
            if slope[0] >= 0:
                x = np.linspace(0,600,5)

            else:
                x = np.linspace(0,-600,5)

            y = slope[0]*x + slope[1]

        else:
            y = np.linspace(0,Y_limit,5)
            x = np.zeros(5)
        
        print(x)
        print(y)
        ax.text(x[3], y[3], f"R = {R_values[i]}", dict(size=10),bbox=dict(boxstyle="round",ec=(0, 0, 0),fc=(1, 1, 1)))
        plt.plot(x, y)

        i += 1
    
    ax.set_ylim(0, Y_limit)
    return


def CLD_definition(dataframe, UTS = 820, UCS = -490, Life_lines_log = [3,4,5,6,7], plot = True):

    for index, row in dataframe.iterrows():
        if row["smax"] < 0:
            dataframe["amp"][index] = row["smax"] * (1 / row["R-value1"] - 1) /2
        else:
            dataframe["amp"][index] = row["smax"] / 2 * (1 - row["R-value1"])
    
    #Find which R values are available
    R_values = list(dataframe.groupby("R-value1").groups.keys())
    print("R values available: ", R_values)

    parameter_dictionary = separateDataFrame(dataframe, separationParameters= ["R-value1", "Temp.", "Cut angle "], separationRanges=[False, [0,40], False]) 
    
    SN_models = []

    for key, df in parameter_dictionary["R-value1"].items(): # go through the dataframe for each r-value
        df = pd.merge(df, parameter_dictionary["Temp."][40]) # merge/take overlap of each dataframe with the desired temperature 
        df = pd.merge(df, parameter_dictionary["Cut angle "][0.0]) # merge/take overlap of each dataframe with the desired cut angle 
        SN_models.append(regression(np.array(df["Ncycles"]), np.array(df["amp"])))

    print("Number of regression models available: ", len(SN_models))

        # # DEBUGGING - plot S-N curve for every R
    dftemp = pd.merge(parameter_dictionary["R-value1"][10], parameter_dictionary["Temp."][40]) # merge/take overlap of each dataframe with the desired temperature 
    dftemp = pd.merge(dftemp, parameter_dictionary["Cut angle "][0.0])
    print(dftemp)
    colors = ["tab:orange","tab:green","tab:blue"]
    for valIndex, Rval in enumerate(parameter_dictionary["R-value1"].keys()):
        plt.scatter(pd.merge(parameter_dictionary["R-value1"][Rval], parameter_dictionary["Cut angle "][0.0])["Ncycles"], (np.absolute(pd.merge(parameter_dictionary["R-value1"][Rval],parameter_dictionary["Cut angle "][0.0])["amp"])), c=colors[valIndex])
        x1 = np.linspace(0,10)
        x2 = np.power(10,SN_models[valIndex].predict(x1.reshape(-1,1)))
        plt.plot(x1, x2, label = Rval, c=colors[valIndex])
    plt.legend()

    #------------------------------------------------------------------------------------
    #################### Slope calculation and ordering

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
    print("The list of slopes and intersects for each R line:\n")
    print(R_slopes_coeff)

    #Creates a list to sort the R lines by the inverse of the slope
    List_for_sorting = np.power(R_slopes_coeff[:, 0],-1)

    #Find the indices that would sort the list
    sort_indices = np.argsort(List_for_sorting)

    #Sort the R_values , R_slopes_coeff and the SN_models
    R_slopes_coeff = R_slopes_coeff[sort_indices]
    SN_models = np.array(SN_models)[sort_indices]
    R_values = np.array(R_values)[sort_indices]


    #------------------------------------------------------------------------------------
    #################### Creation of constant life lines ands R lines
    fig, ax = plt.subplots()
    R_line_visualizer(R_slopes_coeff,R_values,ax)

    #------------------- Create constant life lines
    amp_plot_lists = []
    mean_plot_lists = []

    for life in Life_lines_log:
        amp_list = []
        mean_list = []

        mean_list.append(UCS)
        amp_list.append(0)

        for i in range(len(SN_models)):
            #----Add STC and STT points

            amp = 10**(float(SN_models[i].predict(np.array(life).reshape(-1, 1))))
            mean = convert_to_mean_stress(amp,R_values[i])
            amp_list.append(amp)
            mean_list.append(mean)
        
        mean_list.append(UTS)
        amp_list.append(0)

        amp_plot_lists.append(amp_list)
        mean_plot_lists.append(mean_list)

    if plot:
        for p in range(len(amp_plot_lists)):
            ax.plot(mean_plot_lists[p], amp_plot_lists[p], label=f"N = 10^{Life_lines_log[p]}")
            ax.legend()

        ax.set_xlabel("Mean Stress")
        ax.set_ylabel("Stress Amplitude")

        #------------------ Visualize the CLD graph
        plt.show()

    return R_values, R_slopes_coeff, SN_models, ax

# CLD_definition("CurveModelling/Data/altdata.csv")

def Location_of_target(target_stress_amplitude,target_mean_stress,R_values,R_slopes_coeff):
    #------------------------------------------------------------------------------------
    #################### Prediction of the fatigue life for a given stress amplitude and mean stress

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
        Above = True
        return R_values[min_index],Above

    else:
        print(f"Target is below R = {R_values[min_index]}")
        Above = False
        return R_values[min_index],Above