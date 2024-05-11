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

#################################################
#Settings
#################################################
R_value_to_plot = 0.1
std_num         = 1 # number of standard deviations for uncertainty
#################################################
#Regression prediction
#################################################

#Obtain the data and regression the chosen R value
dataframe = pd.read_csv("CurveModelling/Data/data42alt.csv")
CLD_definition.add_amplitudecol(dataframe)
R_values, R_slopes_coeff, SN_models, parameter_dictionary, std = CLD_definition.CLD_definition(dataframe)
R_index             = np.where(R_values == R_value_to_plot)[0][0]
Reg_model_to_plot   = SN_models[R_index]
Data_to_plot        = parameter_dictionary["R-value1"][R_value_to_plot]
std_to_plot         = std[R_index]*std_num

print(Data_to_plot)

#Create the regression curve points
n_list_reg         = np.linspace(0,10)
amp_list_reg       = np.power(10,Reg_model_to_plot.predict(n_list_reg.reshape(-1,1)))
amp_list_reg_upper = np.power(10,Reg_model_to_plot.predict(n_list_reg.reshape(-1,1)) + std_to_plot)
amp_list_reg_lower = np.power(10,Reg_model_to_plot.predict(n_list_reg.reshape(-1,1)) - std_to_plot)

####################################################
#PINN prediction
####################################################



####################################################
#Plotting
####################################################


#Get the pinn model trained on all R values
fig, ax = plt.subplots()

ax.fill_between(np.power(10,n_list_reg),amp_list_reg_upper, amp_list_reg_lower, alpha=0.5, label ="Confidence interval with " + str(std_num) + " STD", color = "orange")
ax.plot(np.power(10,n_list_reg), amp_list_reg, label ="Basquin Regression R = " + str(R_value_to_plot), color = "blue")
ax.scatter(np.power(10,parameter_dictionary["R-value1"][R_value_to_plot]["Ncycles"]), parameter_dictionary["R-value1"][R_value_to_plot]["amp"], label ="Optidat datapoints ", color = "gray")
ax.set_xscale('log')
ax.set_xlabel('Number of Cycles')
ax.set_ylabel('Amplitude Stress')
ax.legend()

plt.show()