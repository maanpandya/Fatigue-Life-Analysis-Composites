import pandas as pd
from CLD_surface import makeSurface, plotSurface
import CLD_definition
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

#Create a dataframe out of the csv file
dataframe = pd.read_csv("CurveModelling/Data/data42alt.csv")
#Add amplitude column to the dataframe
CLD_definition.add_amplitudecol(dataframe)
#Define the CLD
R_values, R_slopes_coeff, SN_models, parameter_dictionary, std = CLD_definition.CLD_definition(dataframe)

#Plot the CLD

CLD_definition.plot_CLD(R_values, R_slopes_coeff, SN_models, with_bounds=False, std=std, std_num=2)

CLD_definition.plot_regression_models(SN_models, R_values,parameter_dictionary)

surface,x,y,z = makeSurface(R_values,SN_models)

plotSurface(SN_models,R_values,surface,x,y,z)

# #Uncertainty plotting - WIP

# xPlot, yPlot = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
# zPlot = surface(xPlot, yPlot)

# figcld = plt.figure()
# axcld = figcld.add_subplot(111, projection='3d')

# axcld.plot_wireframe(xPlot, zPlot, yPlot, cmap='viridis', alpha=0.5, color="green")
# for i in range(len(SN_models)):
#     axcld.plot(x[i::len(R_values)], z[i::len(R_values)], y[i::len(R_values)])

# axcld.set_xlabel('Mean stress MPa')
# axcld.set_ylabel('log Number of cycles')
# axcld.set_zlabel('Stress amplitude MPa')


# std_num = 2 # number of standard deviation

# #Plot the lower bound - !!! Edits SN_models !!!
# for index, model in enumerate(SN_models):
#     model.intercept_ = model.intercept_ - std[index]*std_num

# CLD_definition.plot_CLD(R_values, R_slopes_coeff, SN_models)

# CLD_definition.plot_regression_models(SN_models, R_values,parameter_dictionary)

# surface,x,y,z = makeSurface(R_values,SN_models)

# # plotSurface(SN_models,R_values,surface,x,y,z)
# zPlot = surface(xPlot, yPlot)

# axcld.plot_wireframe(xPlot, zPlot, yPlot, cmap=cm.Blues, alpha=0.5, color="red")
# for i in range(len(SN_models)):
#     axcld.plot(x[i::len(R_values)], z[i::len(R_values)], y[i::len(R_values)])



# #Plot the upper bound - !!! Edits SN_models !!!
# for index, model in enumerate(SN_models):
#     model.intercept_ = model.intercept_ + std[index]*std_num*2 # 2 because it has to counteract the previous one

# CLD_definition.plot_CLD(R_values, R_slopes_coeff, SN_models)

# CLD_definition.plot_regression_models(SN_models, R_values,parameter_dictionary)

# surface,x,y,z = makeSurface(R_values,SN_models)

# # plotSurface(SN_models,R_values,surface,x,y,z)
# zPlot = surface(xPlot, yPlot)

# axcld.plot_wireframe(xPlot, zPlot, yPlot, cmap=cm.Reds, alpha=0.5, color="blue")
# for i in range(len(SN_models)):
#     axcld.plot(x[i::len(R_values)], z[i::len(R_values)], y[i::len(R_values)])


# #Return SN_ models back to normal
# for index, model in enumerate(SN_models):
#     model.intercept_ = model.intercept_ - std[index]*std_num

plt.show()

def CLD_interpolator_log(surface,amp_stress,R_value):
    mean_stress = CLD_definition.convert_to_mean_stress(amp_stress,R_value)
    return surface(mean_stress,amp_stress)

# Life = CLD_interpolator_log(surface,134,-378)
# print("Amplitude: 134, Mean stress: -378, Life:")
# print(Life)

plt.show()


