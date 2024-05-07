import numpy as np
import matplotlib.pyplot as plt
from blocktestfnct import Calculations
import sys
sys.path.append('NeuralNetworkCode')
import function as f


import CLD_interpolator
from CLD_interpolator import CLD_interpolator_log

surface = CLD_interpolator.surface

code ='RBTE1b2'.lower()   
R = 0.1
Fmax1 = np.arange(30, 70)  # MPa Maximum force of first block
Fmax2 = np.arange(30, 70)  # kN Maximum force of second block
Area = 169.524      #mm^2   Area of specimen
Smax1 = Fmax1 / Area  * (10**3) #MPa
Smax2 = Fmax2 / Area  * (10**3) #MPa
Smax1test = [328.39, 332.74, 337.90, 355.12, 356.22]
Smax2test = [260.61, 263.98, 268.22, 281.89, 282.76]
Resultstest = [10511, 13130, 22500, 980, 7431]

cycles1_array, cycles2_array, x, y = Calculations(Smax1, Smax2, code, surface, R)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, np.log(cycles1_array), rstride=5, cstride=5, linewidth=0.5, color='blue', alpha=0.8, cmap='viridis')
ax.plot_wireframe(x, y, np.log(cycles2_array), rstride=5, cstride=5, linewidth=0.5, color='red', alpha=0.8, cmap='viridis')
ax.scatter(Smax1test, Smax2test, np.log(Resultstest), color='red')
ax.set_xlabel('Smax1 (MPa)')
ax.set_ylabel('Smax2 (MPa)')
ax.set_zlabel('Cycles')
plt.show()


######################################################################################################################################
code ='RBTE1b2'.lower()  

Smax1 = [328.39, 332.74, 337.90, 355.12, 356.22]
Smax2 = [260.61, 263.98, 268.22, 281.89, 282.76]
Results = [10511, 13130, 22500, 980, 7431]



cycles1_array, cycles2_array, x, y = Calculations(Smax1, Smax2, code, surface, R)
cycles1_array = np.diag(cycles1_array)
cycles2_array = np.diag(cycles2_array)
x_values = np.linspace(0, 20, 200)  
y_values = x_values 


plt.scatter(np.log(Results), np.log(cycles1_array), label = "CLD predictions")
plt.scatter(np.log(Results), np.log(cycles2_array), label = "NN predictions")
plt.legend()
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.plot(x_values, y_values, color='red', linestyle='--')
plt.xlabel("Optidat log of number of cycles")
plt.ylabel("CLD and NN log of number of cycles")
plt.show()