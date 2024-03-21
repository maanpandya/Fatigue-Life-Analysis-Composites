import numpy as np
import rainflow 

# Read the file
with open('Wisper/WISPER', 'r') as file:
    file_contents = file.readlines()
array = np.array([float(x) for x in file_contents[1:]])

# Normalize the array to the maximum value
normalized_array = array / np.max(array)

# Count cycles
counted_cycles = rainflow.count_cycles(normalized_array)

# Calculate the accumulated damage
Max_Force = 330 # Mpa

print(counted_cycles)
Accumulated_stress = 0
for i in range (len(counted_cycles)):
    N_Cycle =counted_cycles[i][1] # fatigue life at reference stress 
    Stress = counted_cycles[i][0] * Max_Force# stress
    B = 1.69111*10**22
    N_AI = (Stress/(367.8))**(-1/0.078)
    Accumulated_stress += N_Cycle/N_AI

print(1/Accumulated_stress)
import matplotlib.pyplot as plt

# Calculate N_AI vs stress
stress_values = [counted_cycles[i][0] * Max_Force for i in range(len(counted_cycles))]
N_AI_values = [(Stress / 624.49) ** (-1 / 0.088) for Stress in stress_values]

# Plot N_AI vs stress
plt.plot(N_AI_values, stress_values)  # Changed the order of the axes
plt.xlabel('N_AI')  # Updated the x-axis label
plt.ylabel('Stress')  # Updated the y-axis label
plt.title('N_AI vs Stress')
plt.show()

# Sum up the damage - according to Miner's rule, failure occurs when this sum exceeds 1
