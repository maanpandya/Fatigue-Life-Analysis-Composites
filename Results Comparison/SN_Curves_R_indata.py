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


#################################################
#Settings
#################################################
R_value_to_plot = -1
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



#Get the datapoints from the optidat
datapoints_n_log = np.array(parameter_dictionary["R-value1"][R_value_to_plot]["Ncycles"])
datapoints_amp = np.array(parameter_dictionary["R-value1"][R_value_to_plot]["amp"])

print("There are: ", len(datapoints_n_log), "datapoints for R = ", R_value_to_plot, "in the dataframe selected \n")

#Create the regression curve points
n_list_reg_log        = datapoints_n_log
amp_list_reg          = np.power(10,Reg_model_to_plot.predict(n_list_reg_log .reshape(-1,1)))
amp_list_reg_upper    = np.power(10,Reg_model_to_plot.predict(n_list_reg_log .reshape(-1,1))  + std_to_plot)
amp_list_reg_lower    = np.power(10,Reg_model_to_plot.predict(n_list_reg_log .reshape(-1,1))  - std_to_plot)
n_list_reg            = np.power(10,n_list_reg_log)

####################################################
print("PINN predictions are being calculated \n")
#PINN prediction

#Get the pinn model trained on all R values
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
                    minstress=0, maxstress=600, exp=True, name=name,
                    plot_abs=True, axis=ax, unlog_n=True, amp_s=True, color=None, export_data=True)
y_test = dp.scale(dp.col_filter(pinn_output['expdata'], ['Ncycles'], 'include'), scaler)
x_test = dp.scale(pinn_output['expdata'].drop(columns=['Ncycles', 'R-value1']), scaler)
model_error_dict, pinn_preds = f.test_model(model, scaler, x_test, y_test, plot=False, mute=True)
big_pinn_df = pinn_output['expdata'].join(pinn_preds) # dataframe with all the data for each point in the dataset with R=rvaluetopplot, including pinn inputs(stress and geometry) and pinn predictions
# pinn_output is dictionary with keys: 'preds', 'predn', 'exps', 'expn', 'expdata'
# pinn_preds is dataframe with columns pred_scaled, pred_log, pred and real_scaled, real_log, real
# model_error_dict is a dictionary with keys: 'lMSE', 'lRMSE' (root mean square error), 'lMAE', 'MRE'.
####################################################

####################################################
#NRMSE calculation
####################################################
print("-----------------------------")
#Get the NRMSE of the regression prediction
avg_amp = np.mean(datapoints_amp)
diff_list_reg = amp_list_reg - datapoints_amp
RMSE_reg = ((np.sum(np.square(diff_list_reg)))/len(datapoints_amp))**0.5
NRMSE_reg = RMSE_reg/avg_amp

print("RMSE of the regression prediction: ", RMSE_reg)
print("NRMSE of the regression prediction: ", NRMSE_reg)

#Get the NRMSE of the PINN prediction
print(pinn_preds)

diff_list_PINN = np.array(pinn_preds) - datapoints_amp
RMSE_PINN = ((np.sum(np.square(diff_list_PINN)))/len(datapoints_amp))**0.5
NRMSE_PINN = RMSE_PINN/avg_amp

print("RMSE of the PINN prediction: ", RMSE_PINN)
print("NRMSE of the PINN prediction: ", NRMSE_PINN)

####################################################
#Plotting
####################################################


#Plot the regression curve
ax.fill_between(n_list_reg, amp_list_reg_upper, amp_list_reg_lower, alpha=0.5, label ="Prediction interval with " + str(conf*100) + "% confidence", color = "orange")
ax.plot(n_list_reg, amp_list_reg, label ="Basquin Regression R = " + str(R_value_to_plot), color = "blue")

#Plot the PINN prediction

#Scatter the datapoints
ax.scatter(datapoints_n, datapoints_amp, label ="Optidat datapoints ", color = "gray")
ax.plot(pinn_output['predn'], pinn_output['preds'], label=f'Prediction by PINN, R = {R_value_to_plot}')
ax.set_xscale('log')
ax.set_xlabel('Number of Cycles')
ax.set_ylabel('Amplitude Stress')
ax.set_xlim(1, 10**7)
ax.legend()






plt.show()