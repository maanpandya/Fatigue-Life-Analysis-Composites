import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Add the directory that contains the modules you want to import to sys.path
Curve_modeling_path = os.path.join(os.getcwd(), 'CurveModelling')
sys.path.append(os.path.join(Curve_modeling_path))

#Now you can import your modules as if they were in the same folder
import CLD_definition
from CLD_surface import makeSurface, plotSurface
from Data_processing import separateDataFrame

#################################################
#Settings
#################################################
R_value_to_plot   = 0.1
R_to_remove       = 0.1 # Put 0 if you don't want to remove any R value
max_amp_to_plot   = 300

#################################################
#Get the data from dataframe for R value to plot
#################################################
dataframe = pd.read_csv("CurveModelling/Data/data42alt.csv")
CLD_definition.add_amplitudecol(dataframe)

parameter_dictionary = separateDataFrame(dataframe, separationParameters= ["R-value1"], separationRanges=[False, [0,40], False]) 

#Collect the data from the parameter dictionary that has the R value to plot
Data_to_plot = parameter_dictionary["R-value1"][R_value_to_plot]
Data_Ncycles = Data_to_plot["Ncycles"]
Data_amp     = Data_to_plot["amp"]

#################################################
#CLD prediction 
#################################################

#Remove the R value that wants to be neglected
dataframe = dataframe[dataframe["R-value1"] != R_to_remove]
#Create CLD surface
R_values, R_slopes_coeff, SN_models, parameter_dictionary, std = CLD_definition.CLD_definition(dataframe)
surface,x,y,z       = makeSurface(R_values,SN_models)

#Create CLD points curve points
amp_cld             = np.linspace(0,max_amp_to_plot,200)
mean_cld            = CLD_definition.convert_to_mean_stress(amp_cld,R_value_to_plot)
n_cld   = surface(mean_cld ,amp_cld)

####################################################
#PINN prediction
####################################################



####################################################
#Plotting
####################################################


#Get the pinn model trained on all R values
fig, ax = plt.subplots()
ax.scatter(Data_Ncycles , Data_amp )
ax.plot(n_cld, amp_cld , label ="CLD prediction R = " + str(R_value_to_plot))
ax.legend()

plt.show()