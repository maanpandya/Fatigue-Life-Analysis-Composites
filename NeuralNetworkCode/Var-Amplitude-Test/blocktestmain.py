import numpy as np
import matplotlib.pyplot as plt
from blocktestfnct import Calculations
import sys
sys.path.append('NeuralNetworkCode')
import function as f


import CLD_interpolator
from CLD_interpolator import CLD_interpolator_log

surface = CLD_interpolator.surface


Fmax1 = np.arange(30, 70)  # MPa Maximum force of first block
Fmax2 = np.arange(30, 70)  # kN Maximum force of second block
Area = 169.524      #mm^2   Area of specimen
Smax1 = Fmax1 / Area  * (10**3) #MPa
Smax2 = Fmax2 / Area  * (10**3) #MPa

#General testing data
code = ['BTC1B2'.lower(), 'BTF31B'.lower(), 'RBTD1B3'.lower(), 'RBTE1B2'.lower()]
R = [-1, 0.1, -1, 0.1]
Smax1test = []
Smax2test = []
Resultstest = []
ResultsNN = []
ResultsCLD = []
#TEST 1
Smax1test.append([225.0128139,230.165805,244.0899847,241.1486273,234.0905047,236.3007464,])
Smax2test.append([176.5476394,180.602603,191.535333,189.207952,181.00943,184.4772134])
Resultstest.append([49855-2500,39934-2500,44062-2500,22546-2500,68475-2500,56668-2500])

#TEST 2
Smax1test.append([203.0884705,198.4728235,198.478812,195.460943,191.5176763])
Smax2test.append([345.7610434,337.9028379,337.8606785,332.7468919,324.0756813])
Resultstest.append([501225-500000,503699-500000,504274-500000,501546-500000,500907-500000])

#TEST 3
Smax1test.append([233.1056794,242.2183794,240.329117,226.0538642,230.9340954,227.0480226,229.0764577,227.3982259])
Smax2test.append([133.8365684,139.0810277,137.9712637,129.3911007,132.1664583,129.9435028,131.0872531,130.1180755])
Resultstest.append([232957,106460,80412,311578,278832,231160,231172,314343])


#TEST 4
Smax1test.append([328.3978292,332.7348893,338.1107886,355.1194106,356.221257])
Smax2test.append([260.6182161,263.9736685,268.3841316,281.8851626,282.7669903])
Resultstest.append([10511,14304,22574,5982,7431])



for i in range(4):
    cycles1_array, cycles2_array, x, y = Calculations(Smax1test[i], Smax2test[i], code[i], surface, R[i])
    ResultsCLD.append(np.diag(cycles1_array))
    ResultsNN.append(np.diag(cycles2_array))
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, np.log(cycles1_array), rstride=5, cstride=5, linewidth=0.5, color='blue', alpha=0.8, cmap='viridis')
    ax.plot_wireframe(x, y, np.log(cycles2_array), rstride=5, cstride=5, linewidth=0.5, color='red', alpha=0.8, cmap='viridis')
    ax.scatter(Smax1test[i], Smax2test[i], np.log(Resultstest[i]), color='red')
    ax.set_xlabel('Smax1 (MPa)')
    ax.set_ylabel('Smax2 (MPa)')
    ax.set_zlabel('Cycles')
    plt.show()"""


######################################################################################################################################

x_values = np.linspace(0.9, 10000000, 200)  
y_values = x_values 
colours1 = ["forestgreen", "limegreen", "mediumturquoise", "deepskyblue"]
colours2 = ["red", "coral", "yellow", "darkorange"]

print("results")
print(Resultstest)
print(ResultsCLD)
print(ResultsNN)
for i in range(4):
    plt.scatter(Resultstest[i], ResultsCLD[i], label=f"CLD predictions {code[i]}",  alpha=0.8, marker ="o")
    plt.scatter(Resultstest[i], ResultsNN[i], label=f"NN predictions {code[i]}",  alpha=0.8, marker ="D")

plt.xlabel("log(Resultstest)")
plt.ylabel("log(Results)")
plt.legend()

plt.xlim(0.9, 1000000)
plt.ylim(0.9, 1000000)
plt.xscale("log")
plt.yscale("log")
plt.plot(x_values, y_values, color='red', linestyle='--')
plt.xlabel("Optidat of number of cycles")
plt.ylabel("CLD and NN of number of cycles")
plt.show()