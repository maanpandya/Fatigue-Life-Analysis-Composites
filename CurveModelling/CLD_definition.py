import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from Data_processing import separateDataFrame
from SNCurve import regression
import random as rd

# pd.options.mode.chained_assignment = None
# np.set_printoptions(precision=4)

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
    Y_limit = 450
    Y_base = 250
    i = 0
    for slope in R_slopes_coeff:
        if R_values[i] != -1:
            if slope[0] >= 0:
                x = np.linspace(0,Y_base/abs(slope[0]) + 50,5)

            else:
                x = np.linspace(0,-Y_base/abs(slope[0]) - 50,5)

            y = slope[0]*x + slope[1]

        else:
            y = np.linspace(0,Y_limit,5)
            x = np.zeros(5)

        ax.text(x[3], y[3], f"R = {R_values[i]}", dict(size=10),bbox=dict(boxstyle="round",ec=(0, 0, 0),fc=(1, 1, 1)))
        plt.plot(x, y)

        i += 1
    
    ax.set_ylim(0, Y_limit)
    return


def add_amplitudecol(dataframe):
    dataframe["amp"] = 0.
    for index, row in dataframe.iterrows():
        if row["smax"] < 0:
            dataframe.loc[index, "amp"] = row["smax"] * (1 / row["R-value1"] - 1) /2
        else:
            dataframe.loc[index, "amp"] = row["smax"] / 2 * (1 - row["R-value1"])
    return dataframe


def CLD_definition(dataframe):
    print("-----------------------------")
    print("Running CLD definition...")
    print("\n")
    
    #Find which R values are available
    R_values = list(dataframe.groupby("R-value1").groups.keys())
    print("The dictionary contains the following R values: ")
    print(R_values)

    parameter_dictionary = separateDataFrame(dataframe, separationParameters= ["R-value1"], separationRanges=[False, [0,40], False]) 
    
    SN_models = []
    pbound = []
    from SNCurve import predband

    i = 0
    for key, df in parameter_dictionary["R-value1"].items(): # go through the dataframe for each r-value
        # df = pd.merge(df, parameter_dictionary["Temp."][40]) # merge/take overlap of each dataframe with the desired temperature 
        # df = pd.merge(df, parameter_dictionary["Cut angle "][0.0]) # merge/take overlap of each dataframe with the desired cut angle 
        SN_models.append(regression(np.array(df["Ncycles"]), np.array(df["amp"])))
        pbound.append(predband(np.array(df["Ncycles"]), np.array(df["amp"]), SN_models[i]))
        i += 1
 
    print("Number of regression models available: ", len(SN_models))

    #     # # DEBUGGING - plot S-N curve for every R
    # # dftemp = pd.merge(parameter_dictionary["R-value1"][10], parameter_dictionary["Temp."][40]) # merge/take overlap of each dataframe with the desired temperature 
    # # dftemp = pd.merge(dftemp, parameter_dictionary["Cut angle "][0.0])
    # # print(dftemp)
    # colors = ['#2ca02c','#d62728','#9467bd', '#8c564b','#e377c2','#1f77b4','#ff7f0e','#7f7f7f', '#bcbd22', '#17becf']

    # for valIndex, Rval in enumerate(parameter_dictionary["R-value1"].keys()):
    #     plt.scatter(parameter_dictionary["R-value1"][Rval]["Ncycles"], parameter_dictionary["R-value1"][Rval]["amp"], c=colors[valIndex])
    #     x1 = np.linspace(0,10)
    #     x2 = np.power(10,SN_models[valIndex].predict(x1.reshape(-1,1)))
    #     plt.plot(x1, x2, label ="R = " + str(Rval), c=colors[valIndex])
    # plt.legend()

    #------------------------------------------------------------------------------------
    #################### Slope calculation and ordering

    #Define intersection point for the slopes
    N_to_connect = 10**3
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
    pbound = np.array(pbound)[sort_indices]

    print("\nCLD is fully defined.")
    print("-----------------------------\n")
    return R_values, R_slopes_coeff, SN_models, parameter_dictionary, pbound


def plot_regression_models(SN_models, R_values, parameter_dictionary):
    colors = ['#2ca02c','#d62728','#9467bd', '#8c564b','#e377c2','#1f77b4','#ff7f0e','#7f7f7f', '#bcbd22', '#17becf']

    fig, ax = plt.subplots()

    for valIndex, Rval in enumerate(R_values):
        ax.scatter(parameter_dictionary["R-value1"][Rval]["Ncycles"], parameter_dictionary["R-value1"][Rval]["amp"], c=colors[valIndex])
        x1 = np.linspace(0,10)
        x2 = np.power(10,SN_models[valIndex].predict(x1.reshape(-1,1)))
        ax.plot(x1, x2, label ="R = " + str(Rval), c=colors[valIndex])
    ax.legend()

    return


def make_life_lines(fig, ax, R_values, R_slopes_coeff, SN_models, Life_lines_log = [3,4,5,6,7], UTS = 820, UCS = -490, x = np.linspace(-800,800)):
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

    y = []
    for life in range(len(mean_plot_lists)):
        y.append(np.interp(x, mean_plot_lists[life], amp_plot_lists[life]))

    return y

def plot_CLD(R_values, R_slopes_coeff, SN_models, Life_lines_log = [3,4,5,6,7], UTS = 820, UCS = -490, with_bounds = False, pbound =[]):

    #################### Creation of constant life lines ands R lines
    fig, ax = plt.subplots()
    R_line_visualizer(R_slopes_coeff,R_values,ax)

    #------------------- Create constant life lines
    colors = list(mcolors.TABLEAU_COLORS.values()) + list((mcolors.BASE_COLORS.values()))
    print(colors)
    cx = np.linspace(UCS,UTS, 200)
    cy = make_life_lines(fig, ax, R_values, R_slopes_coeff, SN_models, Life_lines_log, UTS, UCS, cx)
    for life in range(len(Life_lines_log)):
        ax.plot(cx, cy[life], label=f"N = 10^{Life_lines_log[life]}", color=colors[life+len(R_values)])
    ax.legend()

    if with_bounds:
        for index, model in enumerate(SN_models):
            model.intercept_ = model.intercept_ - pbound[index]
        cyl = make_life_lines(fig, ax, R_values, R_slopes_coeff, SN_models, Life_lines_log, UTS, UCS, cx)

        for index, model in enumerate(SN_models):
            model.intercept_ = model.intercept_ + pbound[index]*2 # 2 because it has to counteract the previous one
        cyu = make_life_lines(fig, ax, R_values, R_slopes_coeff, SN_models, Life_lines_log, UTS, UCS, cx)

        for index, model in enumerate(SN_models):
            model.intercept_ = model.intercept_ - pbound[index] # return models back to normal

        for life in range(len(Life_lines_log)):
            ax.fill_between(cx, cyl[life], cyu[life], alpha = 0.2, color=colors[life+len(R_values)])

    ax.set_xlabel("Mean Stress")
    ax.set_ylabel("Stress Amplitude")
    
    return ax


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
    