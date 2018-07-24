'''
backprojection_mk3.py
@author   Kiryu Sakakibara, Allen Wang, Charles Cheng, Ethan Fang
@since    7/16/2018
Using the data of signal pulses and UAS coordinates, forms an image using back-projection
'''
import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
import timeit

start = timeit.default_timer()

f = open(r"D:\Desktop\UAV-SAR\mandrill_no_aliasing_data.pkl", "rb")
data = pickle.load(f)

size = 500 # number of pixels per axis of the image, change to alter resolution

'''
Returns the distance(range) between the radar and a specific location in the 3D cartesian coordinate plane
'''
def get_range(radar_xpos, radar_ypos, radar_zpos, img_x, img_y):
    return math.sqrt((radar_xpos - img_x)**2 + (radar_ypos - img_y)**2 + (radar_zpos)**2)

'''
Unfinished
'''
def edge_detection(arr):
    return

'''
Stores the positions of the UAS
'''
radar_x = []
radar_y = []
radar_z = []
for position in data[0]:
    radar_x.append(position[0])
    radar_y.append(position[1])
    radar_z.append(position[2])

'''
pulses = data[1]
range_bins = data[2][0]
range_bin_d = (range_bins[int(len(range_bins)/2)+1] - range_bins[int(len(range_bins)/2)])/2
...

Initializes grid as a list of lists
'''
pixel_values = list(list(0+0j for ii in np.arange(0, size)) for jj in np.arange(0, size))

'''
Calculating the color of each pixel in the image by iterating over all the pulses and summing the complex values within
the correct range bins and ultimately storing the absolute value of the sum at the corresponding index in pixel_values 
'''
y = 2.5
for ii in np.arange(0, size):
    print ("%d / %d" % (ii, size))
    x = -2.5
    for jj in np.arange(0, size):
        for kk in np.arange(0, 100):
            distance = get_range(radar_x[kk], radar_y[kk], radar_z[kk], x, y)
            ratio = (distance % range_bin_d) / range_bin_d
            index = math.floor(distance/range_bin_d)
            pixel_values[ii][jj] += (pulses[kk][index]*(1-ratio) + pulses[kk][index+1]*(ratio))
        pixel_values[ii][jj] = np.abs(pixel_values[ii][jj])
        x = x + 5.0/size
    y = y - 5.0/size
          
plt.imshow(pixel_values) # shows the image

end = timeit.default_timer()
print(end-start) # prints the run time
