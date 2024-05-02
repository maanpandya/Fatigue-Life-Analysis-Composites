import math
import numpy as np
import matplotlib.pyplot as plt
from blocktestfnct import Calculations
import numpy as np
import sys
sys.path.append('NeuralNetworkCode')

import numpy as np
import matplotlib.pyplot as plt
import function as f
import torch
import pandas as pd



code ='RBTE1b2'.lower()        

Fmax1 = np.arange(68, 70)  # MPa Maximum force of first block
Fmax2 = np.arange(68, 70)  # kN Maximum force of second block

Area = 169.524      #mm^2   Area of specimen
Smax1 = Fmax1 / Area  * (10**3) #MPa
Smax2 = Fmax2 / Area  * (10**3) #MPa
#STDEV1 = 
#STDEV2 = 



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot wireframe for cycles1_array
ax.plot_wireframe(x, y, np.log(cycles1_array), rstride=5, cstride=5, linewidth=0.5, color='blue', alpha=0.8, cmap='viridis')

# Plot wireframe for cycles2_array
ax.plot_wireframe(x, y, np.log(cycles2_array), rstride=5, cstride=5, linewidth=0.5, color='red', alpha=0.8, cmap='viridis')

ax.scatter(356.215341339503, 282.762294443245, np.log(650+6781), color='red')
ax.set_xlabel('Smax1 (MPa)')
ax.set_ylabel('Smax2 (MPa)')
ax.set_zlabel('Cycles')
plt.show()


######################################################################################################################################
code ='RBTE1b2'.lower()        


Area = 169.524      #mm^2   Area of specimen
Smax1 = [328.39, 332.74, 337.90, 355.12, 356.22]
Smax2 = [260.61, 263.98, 268.22, 281.89, 282.76]
Results = [10511, 13130, 22500, 980, 7431]



#STDEV1 = 
#STDEV2 = 

letter_lst = []
for letter in code:
    letter_lst.append(letter)

if letter_lst[0] == 'b' : #normal block test
    if letter_lst[3] == '1' :
        N1 = 2500
    if letter_lst[3] == '2' :
        N1 = 25000
    if letter_lst[3] == '3' :
        N1 = 5e5
    #number of cycles till damage = 1
    N2 = []
    for i in range(len(Smax2)):
        for k in range(len(Smax1)):
            N2.append(SNcurve2(Smax2[i]) * ( 1 - (N1 / SNcurve1(Smax1[k]))))
    #deltaN2 = STDEV2 + (SNcurve2(Smax2) * N1 / SNcurve1(Smax1)) * (STDEV2/SNcurve2(Smax2) + STDEV1/SNcurve1(Smax1)))
            print(f'The material tested with {code} experiences {N2[i]}  cycles at a first load level of {Smax2[i]} MPa and a second load level of {Smax2[i]} MPa')
    """print(SNcurve2(Smax2))
    print(SNcurve1(Smax1))
    print(N1 / SNcurve1(Smax1))
    print(N1)"""


if letter_lst[0] == 'r' : #repeated block test
    if letter_lst[4] == '1' :
        N1 = 25
        if letter_lst[6] == '1' :
            N2 = 25
        if letter_lst[6] == '2' :
            N2 = 250
        if letter_lst[6] == '3' :
            N2 = 5000
    if letter_lst[4] == '2' :
        N1 = 250
    if letter_lst[4] == '3' :
        N1 = 5000
    if letter_lst[5] == '1' :
        N2 = 25
    if letter_lst[5] == '2' :
        N2 = 250
    if letter_lst[5] == '3' :
        N2 = 5000
    cycles1 =  np.zeros((len(Smax2), len(Smax2)))
    
    cycles2 =  np.zeros((len(Smax2), len(Smax2)))

    for i in range(len(Smax2)):
        for k in range(len(Smax1)):
            damage1 = N1 / SNcurve1(Smax1[k], 0) + N2 / SNcurve2(Smax2[i],0)
            damage2 = N1 / NNmodel(Smax1[k]) + N2 / NNmodel(Smax2[i])
            cycles1[i][k] = (1 / damage1)*(N1+N2)
            cycles2[i][k] = (1 / damage2)*(N1+N2)

            print("This is the iteration number")
            print((i+1)*(k+1))
            print("The damage")
            print(damage1)
            print("and")
            print(damage2)
    # Convert Smax1 and Smax2 to numpy arrays for plotting
    Smax1_array = np.array(Smax1)
    print(Smax1_array)
    Smax2_array = np.array(Smax2)
    print(Smax2_array)

    x, y = np.meshgrid(Smax1_array, Smax2_array)
    cycles1_array = np.diag(np.array(cycles1))
    cycles2_array = np.diag(np.array(cycles2))


    print(cycles1_array)
    print(cycles2_array)

print(Results)
print(cycles1_array)

plt.scatter(np.log(Results), np.log(cycles1_array), label = "CLD predictions")
plt.scatter(np.log(Results), np.log(cycles2_array), label = "NN predictions")
plt.legend()

plt.xlim(0, 11)
plt.ylim(0, 11)
x_values = np.linspace(0, 10, 200)  # Creating an array of x values from 0 to 5
y_values = x_values  # Since x = y, y values will be the same as x values
plt.plot(x_values, y_values, color='red', linestyle='--')
plt.show()
plt.show()