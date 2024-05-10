import math  
import numpy as np
import rainflow 
import matplotlib.pyplot as plt
import pandas as pd
from   NNfuncion import NNfuntion 
import CLD_interpolator
from CLD_interpolator import CLD_interpolator_log



def SNfunction(normalized_array, wawas, surface):  
    count_2 = 0
    rng_list = []
    R_list = []
    xd  = []
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(normalized_array):
        count_2 += 1 
        rng_list.append(rng)
        R = (mean- rng/2) / (mean + rng/2) 
        R_list.append(R)
        xd.append(count_2)

    R_counted = []
    Rng_counted = []
    n_times_apeared = []
    for i in range(172):
        n_times_apeared.append(0)
    appered = 0 
    j = 0

    for i in range(len(rng_list)):
            appered = 0
            for j in range(len(R_counted)):
                if rng_list[i] == Rng_counted[j] and R_list[i] == R_counted[j] and appered == 0:
                    n_times_apeared[j] = n_times_apeared[j] +1
                    appered = 1
            if appered == 0:
                R_counted.append(R_list[i])
                n_times_apeared[j] = n_times_apeared[j] +1
                Rng_counted.append(rng_list[i])
    cycles = []
    stresses = []
    for j in wawas:
        Accumulated_stress = 0
        for i in range(len(R_counted)):
            N_Cycle =n_times_apeared[i] # fatigue life at reference stress 
            Rng_counted[i]*j# stress x*(1-R_value)/2
            N_SN = CLD_interpolator_log(surface,Rng_counted[i]*j*(1-R_counted[i])/2, R_counted[i])
            Accumulated_stress += N_Cycle/N_SN
        cycles.append(1/Accumulated_stress)
        stresses.append(j)

    
    return cycles, stresses

