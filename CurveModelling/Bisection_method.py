from CLD_plotter import Amplitude_finder
import numpy as np

def mid_point(val1,val2):
        mid = (val2-val1)/2
        return mid     

def orthogonal_projection(point1,point2,target):
        #project the target point on the line between midpoints and returns the distance to the line#

        target = np.array(target)
        point1 = np.array(point1)
        point2 = np.array(point2)

        line_dir = point2 - point1
        target_vec = target - point1
        projection_scalar = np.dot(target_vec, line_dir) / np.dot(line_dir, line_dir)
        projection_vector = projection_scalar * line_dir
        distance_vec = target - (point1 + projection_vector)
        distance = np.linalg.norm(distance_vec)


        # positive direction indicates that the point is above the line
        cross_product = np.cross(line_dir,distance_vec)
        if cross_product > 0:
                distance *= -1
        
        return distance

def interval_bisection(a,b,c,d):
        epsilon = 1
        mid_ab = (mid_point(a[0],b[0]),mid_point(a[1],b[1]))
        mid_cd = (mid_point(c[0],d[0]),mid_point(c[1],d[1]))
        for i in range(1000):
            distance = orthogonal_projection(mid_ab, mid_cd)
            if distance <= epsilon:
                return mid_ab, mid_cd
            elif distance > 0:
                mid_ab = mid_ab + (b-a)
                mid_cd = mid_cd + (d-c)
            else:
                mid_ab = mid_ab - (b-a)
                mid_cd = mid_cd - (d-c)
        return mid_ab, mid_cd

target = (100,220) # (mean,amplitude)
target_R_value = (target[0]-target[1])/(target[0]+target[1])

if (target_R_value < 10) and (target_R_value > -1):
        a = (0,Amplitude_finder(-1,10**5))
        b = (0,Amplitude_finder(-1,10**4))
        c = (Amplitude_finder(0.1,10**5)*(0.1+1)/(1-0.1),Amplitude_finder(0.1,10**5))
        d = (Amplitude_finder(0.1,10**4)*(0.1+1)/(1-0.1),Amplitude_finder(0.1,10**4))

print(interval_bisection(a,b,c,d))