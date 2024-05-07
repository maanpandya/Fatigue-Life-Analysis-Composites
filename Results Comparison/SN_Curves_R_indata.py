import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get the current directory
current_directory = os.getcwd()
# Add the directory that contains the modules you want to import to sys.path
sys.path.append(os.path.join(current_directory, 'CurveModelling'))

# Now you can import your modules as if they were in the same folder
import CLD_definition

#################################################
#Define which R value to plot
#################################################
R_value_to_plot = 10

#################################################
#Obtain the data and model for the chosen R value
#################################################

dataframe = pd.read_csv("CurveModelling/Data/data42alt.csv")
CLD_definition.add_amplitudecol(dataframe)
R_values, R_slopes_coeff, SN_models,parameter_dictionary,std = CLD_definition.CLD_definition(dataframe)

R_index = np.where(R_values == R_value_to_plot)[0][0]
Reg_model_to_plot = SN_models[R_index]
Data_to_plot = parameter_dictionary["R-value1"][R_value_to_plot]
std_to_plot = std[R_index]

print(Data_to_plot)

######################################
#Get the pinn model 
######################################



######################################
#Get the pinn model 
######################################


fig, ax = plt.subplots()

ax.scatter(parameter_dictionary["R-value1"][R_value_to_plot]["Ncycles"], parameter_dictionary["R-value1"][R_value_to_plot]["amp"])
n_list_reg = np.linspace(0,10)
amp_list_reg = np.power(10,Reg_model_to_plot.predict(n_list_reg.reshape(-1,1)))
amp_list_reg_upper = np.power(10,Reg_model_to_plot.predict(n_list_reg.reshape(-1,1)) + std_to_plot)
amp_list_reg_lower = np.power(10,Reg_model_to_plot.predict(n_list_reg.reshape(-1,1)) - std_to_plot)

ax.plot(n_list_reg, amp_list_reg, label ="Basquin Regression R = " + str(R_value_to_plot))
ax.fill_between(n_list_reg, amp_list_reg_upper, amp_list_reg_lower, alpha=0.5)
ax.legend()

plt.show()



# R_to_plot = R_values[R_index]
# SN_to_plot = SN_models[R_index]



# fig, ax = plt.subplots()

# for valIndex, Rval in enumerate(R_values):
#     ax.scatter(parameter_dictionary["R-value1"][Rval]["Ncycles"], parameter_dictionary["R-value1"][Rval]["amp"], c=colors[valIndex])
#     x1 = np.linspace(0,10)
#     x2 = np.power(10,SN_models[valIndex].predict(x1.reshape(-1,1)))
#     ax.plot(x1, x2, label ="R = " + str(Rval), c=colors[valIndex])
# ax.legend()
