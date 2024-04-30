import scipy as sp
import numpy as np
import pandas as pd

import CLD_definition

#Create a dataframe out of the csv file
dataframe = pd.read_csv("CurveModelling/Data/altdata.csv")
CLD_definition.add_amplitudecol(dataframe)

R_values, R_slopes_coeff, SN_models, ax = CLD_definition.CLD_definition(dataframe, plot=False)

values = [x/10. for x in range(1,10)]
coords = np.zeros((len(values),2))

amp_plot_lists = []
mean_plot_lists = []

for life in values:
    amp_list = []
    mean_list = []

    mean_list.append(-490)
    amp_list.append(0)

    for i in range(len(SN_models)):
        #----Add STC and STT points

        amp = 10**(float(SN_models[i].predict(np.array(life).reshape(-1, 1))))
        mean = CLD_definition.convert_to_mean_stress(amp,R_values[i])
        amp_list.append(amp)
        mean_list.append(mean)
    
    mean_list.append(820)
    amp_list.append(0)

    amp_plot_lists.append(amp_list)
    mean_plot_lists.append(mean_list)

coords = mean_plot_lists
