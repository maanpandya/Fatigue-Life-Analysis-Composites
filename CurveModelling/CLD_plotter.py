import numpy as np
import matplotlib.pyplot as plt
from Data_processing import separateDataFrame
#Import function to find the regression courve for the given load ratio
np.set_printoptions(precision=3)


def Amplitude_finder(R, N, PINN = "False"):
    """
    This function finds the amplitude of the stress for a given load ratio, mean stress and number of cycles
    """
    #Use function to separate take the stress amplitude and number of cycles for the given load ration from the pandas dataframe
    #Use function to find the regression curve for the given load ratio, x axis is number of cycles and y axis is amplitude
    #Use regression model to find the amplitude for the given number of stress cycles and load ratio


    #-----R = -1,
    if R == -1 and N == 10**7:
        Amp = 70
    
    if R == -1 and N == 10**3:
        Amp = 265

    if R == -1 and N == 10**4:
        Amp = 190

    if R == -1 and N == 10**5:
        Amp = 135
    
    #-----R = 10
    if R == 10 and N == 10**7:
        Amp = 95

    if R == 10 and N == 10**3:
        Amp = 150
    
    if R == 10 and N == 10**4:
        Amp = 140

    if R == 10 and N == 10**5:
        Amp = 125


    #-----R = 0.1
    if R == 0.1 and N == 10**7:
        Amp = 65

    if R == 0.1 and N == 10**3:
        Amp = 180
    
    if R == 0.1 and N == 10**4:
        Amp = 140
    
    if R == 0.1 and N == 10**5:
        Amp = 105

    return Amp

def Mean_stress_finder(Amp,R):
    """
    This function finds the mean stress for a given load ratio, amplitude and number of cycles
    """
    S_max = Amp*2/(1-R)
    S_min = S_max*R
    Mean = (S_max + S_min)/2

    return Mean

#------------------- Select which load ratios and number of cycles to plot
N_lines = [10**3,10**4,10**7]
R_list = [10,
          -1,
          0.1]

print("Curves of number of cycles to failure in log:\n",np.log10(N_lines))
print("Load ratio to be studied:\n",R_list)

#------------------- Find amplitude for each load ratio
"""
Runs a loop for each number of cycles to failure for every load ratio
Creates a list of data points with the following sequence:
[Ratio, Amplitude, Mean stress, Number of cycles]
"""

Data_points = []

for N in N_lines:
    for ratio in R_list:
        amp = Amplitude_finder(ratio, N)
        mean = Mean_stress_finder(amp, ratio)
        Data_points.extend([ratio, amp, mean, N])

Data_points = np.array(Data_points).reshape(len(N_lines)*len(R_list),4)

print(Data_points)

#------------------- Plot CLD: mean stress on x axis and amplitude on y axis
"""
Separates the data points for each number of cycles to failure
Plots a curve for each number of cycles to failure
"""

fig, ax = plt.subplots()

#Plot the first constant life line
amp_plot = Data_points[0:len(R_list),1]
mean_plot = Data_points[0:len(R_list),2]
plt.plot(mean_plot,amp_plot)
print(amp_plot)

#Plot the remaining constant life lines
for n in range(1,len(N_lines)):
    amp_plot = Data_points[n*len(R_list):(n+1)*len(R_list),1]
    mean_plot = Data_points[n*len(R_list):(n+1)*len(R_list),2]

    plt.plot(mean_plot,amp_plot)
    print(amp_plot)

ax.set_ylim(bottom=0)
plt.show()