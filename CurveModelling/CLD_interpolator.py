import pandas as pd
from CLD_surface import makeSurface, plotSurface
import CLD_definition
import matplotlib.pyplot as plt


#Create a dataframe out of the csv file
dataframe = pd.read_csv("CurveModelling/Data/data42alt.csv")
#Add amplitude column to the dataframe
CLD_definition.add_amplitudecol(dataframe)
#Define the CLD
R_values, R_slopes_coeff, SN_models,parameter_dictionary = CLD_definition.CLD_definition(dataframe)
#Plot the CLD

CLD_definition.plot_CLD(R_values, R_slopes_coeff, SN_models)

CLD_definition.plot_regression_models(SN_models, R_values,parameter_dictionary)

surface,x,y,z = makeSurface(R_values,SN_models)

plotSurface(SN_models,R_values,surface,x,y,z)

def CLD_interpolator_log(surface,amp_stress,R_value):
    mean_stress = CLD_definition.convert_to_mean_stress(amp_stress,R_value)
    return surface(mean_stress,amp_stress)

Life = CLD_interpolator_log(surface,50,0.5)
print("Amplitude: 134, Mean stress: -378, Life:")
print(Life)


plt.show()


