import os
import sys
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

#Add the directory that contains the modules you want to import to sys.path
if os.getcwd()[-10:-1] != 'Composite':
    os.chdir(os.getcwd()[0:-18])
Curve_modeling_path = os.path.join(os.getcwd(), 'CurveModelling')
sys.path.append(os.path.join(Curve_modeling_path))
nn_path = os.path.join(os.getcwd(), 'NeuralNetworkCode')
sys.path.append(os.path.join(nn_path))

#Now you can import your modules as if they were in the same folder
import CLD_definition
import function as f
import DataProcessing.DPfunctions as dp
import CLD_definition
import SNCurve
from CLD_surface import makeSurface, plotSurface
from Data_processing import separateDataFrame

#################################################
#Settings
#################################################
R_value_to_plot   = 0.1
R_to_remove       = 0.1 # Put 0 if you don't want to remove any R value
max_amp_to_plot   = 300
conf              = 0.95 # confidence level for uncertainty
fig, ax = plt.subplots()

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
R_values, R_slopes_coeff, SN_models, parameter_dictionary = CLD_definition.CLD_definition(dataframe)
surface,x,y,z       = makeSurface(R_values,SN_models)

#Create CLD points curve points
amp_cld   = np.linspace(0,max_amp_to_plot,200)
mean_cld  = CLD_definition.convert_to_mean_stress(amp_cld,R_value_to_plot)
n_cld     = surface(mean_cld ,amp_cld)
predband = SNCurve.predband(dataframe["R-value1"], datapoints_amp,Reg_model_to_plot, conf, n_list_reg_log)

#Get the lower surface 
for index, model in enumerate(SN_models):
    model.intercept_ = model.intercept_ - np.mean(std[index]*std_num) 
print("Replace patchwork fix") # MEAN IS PATCHWORK FIX, CHANGE IT LATER !!!!!!
lower_surface, xl, yl, zl = makeSurface(R_values,SN_models, dy = predband)
n_cld_l = lower_surface(mean_cld,amp_cld)

#Get the upper surface 
for index, model in enumerate(SN_models):
    model.intercept_ = model.intercept_ + np.mean(std[index]*std_num)*2 # 2 because it has to counteract the previous one
print("Replace patchwork fix") # MEAN IS PATCHWORK FIX, CHANGE IT LATER !!!!!!
upper_surface, xu, yu, zu = makeSurface(R_values,SN_models)
n_cld_u = upper_surface(mean_cld,amp_cld)

#Return SN_ models back to normal
for index, model in enumerate(SN_models):
    model.intercept_ = model.intercept_ - np.mean(std[index])*std_num 
print("Replace patchwork fix") # MEAN IS PATCHWORK FIX, CHANGE IT LATER !!!!!!

####################################################
#PINN prediction
####################################################
#PINN prediction
path = 'NeuralNetworkCode/NNModelArchive/rev4/pinnlossfinale2'
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
pinn_output = f.complete_sncurve2(datapoint, data, R_value_to_plot, model, scaler,
                    minstress=0, maxstress=600, exp=False, name=name,
                    plot_abs=True, axis=ax, unlog_n=True, amp_s=True, color=None, export_data=True)

####################################################
#Plotting
####################################################

#Get the pinn model trained on all R values
ax.scatter(np.power(10,Data_Ncycles), Data_amp )
ax.plot(np.power(10,n_cld), amp_cld , label ="CLD prediction R = " + str(R_value_to_plot))
ax.plot(np.power(10,n_cld_l), amp_cld , label ="CLD prediction bounds σ = " + str(std_num), c="red")
ax.plot(pinn_output['predn'], pinn_output['preds'], label=f'Prediction by PINN, R = {R_value_to_plot}')
ax.plot(np.power(10,n_cld_u), amp_cld, c="red")
ax.set_xscale('log')
ax.set_xlabel('Number of Cycles')
ax.set_ylabel('Amplitude Stress')
ax.legend()

# Add a comment to the graph saying R_to_remove was removed from the models
ax.text(0.8, 0.7, f"R value = {R_to_remove} \n was removed from models",
        transform=ax.transAxes, ha='center', 
        bbox=dict(facecolor='white', alpha=1, edgecolor='black'),
        fontsize=8)
plt.show()