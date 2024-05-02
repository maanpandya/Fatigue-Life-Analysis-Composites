import scipy as sp
import numpy as np
import pandas as pd

import CLD_definition

#Create a dataframe out of the csv file
dataframe = pd.read_csv("CurveModelling/Data/altdata.csv")
dataframe["amp"] = 0.
R_values, R_slopes_coeff, SN_models, ax = CLD_definition.CLD_definition(dataframe, plot=False)

values = [x/10. for x in range(1,10)]
coords = np.zeros((len(values),2))


