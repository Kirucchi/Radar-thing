# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
import timeit

start = timeit.default_timer()

f = open(r"D:\Desktop\UAV-SAR\mandrill_no_aliasing_data.pkl", "rb")
data = pickle.load(f)

size = 500

def get_range(radar_pos, x, y):
    return math.sqrt((radar_pos - x)**2 + (radar_y-y)**2 + (radar_z)**2)

radar_positions = data[0]
pulses = data[1]
range_bins = data[2][0]

radar_x = []
for position in radar_positions:
    radar_x.append(position[0])
radar_y = radar_positions[0][1]
radar_z = radar_positions[0][2]

pixel_values = list(list(0+0j for ii in np.arange(0, size)) for jj in np.arange(0, size))

y = 2.5
for ii in np.arange(0, size):
    x = -2.5
    for jj in np.arange(0, size):
        for kk in np.arange(0, 100):
            distance = get_range(radar_x[kk], x, y)
            ratio = (distance % 0.0185) / 0.0185
            index = math.floor(distance/0.0185)
            pixel_values[ii][jj] += (pulses[kk][index]*(1-ratio) + pulses[kk][index+1]*(ratio))
        pixel_values[ii][jj] = np.abs(pixel_values[ii][jj])
        x = x + 5.0/size
    y = y - 5.0/size
            
plt.imshow(pixel_values)

end = timeit.default_timer()
print(end-start)
