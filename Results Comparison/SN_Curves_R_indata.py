import os
import sys
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

#Add the directory that contains the modules you want to import to sys.path
if os.getcwd()[-10:-1] != 'Composites':
    os.chdir(os.getcwd()[0:-18])
Curve_modeling_path = os.path.join(os.getcwd(), 'CurveModelling')
sys.path.append(os.path.join(Curve_modeling_path))
nn_path = os.path.join(os.getcwd(), 'NeuralNetworkCode')
sys.path.append(os.path.join(nn_path))

#Now you can import your modules as if they were in the same folder
import CLD_definition
import function as f
import DataProcessing.DPfunctions as dp


#################################################
#Settings
#################################################
R_value_to_plot = -0.4
conf = 0.95 # confidence value - not implemented yet - 95% by default (fraction of data within the bounds)
fig, ax = plt.subplots()
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
std_to_plot         = std[R_index] # *std_num


#Create the regression curve points
n_list_reg         = np.linspace(0,10)
amp_list_reg       = np.power(10,Reg_model_to_plot.predict(n_list_reg.reshape(-1,1)))
amp_list_reg_upper = np.power(10,Reg_model_to_plot.predict(n_list_reg.reshape(-1,1)) + std_to_plot)
amp_list_reg_lower = np.power(10,Reg_model_to_plot.predict(n_list_reg.reshape(-1,1)) - std_to_plot)

####################################################
#PINN prediction
path = 'NeuralNetworkCode/NNModelArchive/rev4/pltest5'
name = path.split('/')[-1]
model, scaler = f.import_model(path)
x_test = dp.dfread(path + '/x_test.csv')
y_test = dp.dfread(path + '/y_test.csv')
data = dp.dfread(path + '/data.csv')
i = rd.choice(data.index)
i = data.index[100]
datapoint = data.loc[i]
print(datapoint)
datapoint = datapoint.to_frame().T
f.complete_sncurve2(datapoint, data, R_value_to_plot, model, scaler,
                    minstress=0, maxstress=600, exp=False, name=name,
                    plot_abs=True, axis=ax, unlog_n=True, amp_s=True, color=None)

####################################################



####################################################
#Plotting
####################################################


#Get the pinn model trained on all R values

ax.fill_between(np.power(10,n_list_reg),amp_list_reg_upper, amp_list_reg_lower, alpha=0.5, label ="Prediction interval with " + str(conf*100) + "% confidence", color = "orange")
ax.plot(np.power(10,n_list_reg), amp_list_reg, label ="Basquin Regression R = " + str(R_value_to_plot), color = "blue")
ax.scatter(np.power(10,parameter_dictionary["R-value1"][R_value_to_plot]["Ncycles"]), parameter_dictionary["R-value1"][R_value_to_plot]["amp"], label ="Optidat datapoints ", color = "gray")
ax.set_xscale('log')
ax.set_xlabel('Number of Cycles')
ax.set_ylabel('Amplitude Stress')
ax.set_xlim(1, 10**7)
ax.legend()

plt.show()