import os
import sys
import pandas as pd
import numpy as np

# Get the current directory
current_directory = os.getcwd()
# Add the directory that contains the modules you want to import to sys.path
sys.path.append(os.path.join(current_directory, 'CurveModelling'))

# Now you can import your modules as if they were in the same folder
import CLD_definition

######################################
#Define which R value to plot
######################################
R_value_to_plot = 0.5

######################################
#Set up the 
######################################

dataframe = pd.read_csv("CurveModelling/Data/data42alt.csv")
CLD_definition.add_amplitudecol(dataframe)
R_values, R_slopes_coeff, SN_models,parameter_dictionary = CLD_definition.CLD_definition(dataframe)

R_index = np.where(R_values == R_value_to_plot)[0][0]
Data_to_plot = parameter_dictionary["R-value1"][R_value_to_plot]

print(Data_to_plot)



# R_to_plot = R_values[R_index]
# SN_to_plot = SN_models[R_index]



# fig, ax = plt.subplots()

# for valIndex, Rval in enumerate(R_values):
#     ax.scatter(parameter_dictionary["R-value1"][Rval]["Ncycles"], parameter_dictionary["R-value1"][Rval]["amp"], c=colors[valIndex])
#     x1 = np.linspace(0,10)
#     x2 = np.power(10,SN_models[valIndex].predict(x1.reshape(-1,1)))
#     ax.plot(x1, x2, label ="R = " + str(Rval), c=colors[valIndex])
# ax.legend()
