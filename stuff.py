from pulson440_unpack import unpack
from backprojection import interp_approach
import matplotlib.pyplot as plt
import numpy as np
import pandas
import math

def extract_platform_position():
    array = list()
    new_array = list()
    position_array = list()
    final_result = list()
    data = pandas.read_csv(platform_position_data, skiprows=2, low_memory = False)
    for elements in data:
        if "Rigid Body" in elements and "Marker" not in elements:
            array.append(elements)
    for contents in array:
        for col in data[contents]:
            if type(col) == str and "Position" in col:
                new_array.append(contents)
    for contents2 in new_array:
        position_array.append(data[contents2][4:].values)
    position_array = np.array(position_array).astype(np.float)
    for contents3 in range(len(position_array[0])):
        mini_array = list()
        for contents4 in range(len(position_array)):
            mini_array.append(position_array[contents4][contents3])
        final_result.append(np.array(mini_array))
    final_result = np.array(final_result)
    return final_result

def extract_given_object():
    array = list()
    new_array = list()
    position_array = list()
    final_result = list()
    data = pandas.read_csv(given_object, skiprows=2, low_memory = False)
    for elements in data:
        if "Rigid Body" in elements and "Marker" not in elements:
            array.append(elements)
    for contents in array:
        for col in data[contents]:
            if type(col) == str and "Position" in col:
                new_array.append(contents)
    for contents2 in new_array:
        position_array.append(data[contents2][4:].values)
    position_array = np.array(position_array).astype(np.float)
    for contents3 in range(len(position_array[0])):
        mini_array = list()
        for contents4 in range(len(position_array)):
            mini_array.append(position_array[contents4][contents3])
        final_result.append(np.array(mini_array))
    final_result = np.array(final_result)
    average = list()
    for indexes in range(3):
        sum = 0
        for elements in range(len(final_result)):
            sum += final_result[elements][indexes]
        average.append(sum/len(final_result))
    return average

def linear_interp_nan(coords, data):
    """
    Linear 1-D interpolation of data that may have missing data and/or 
    coordinates. Assumes that coordinates are uniformly spaced.
    """
    # Initialize outputs; make a deep copy to ensure that inputs are directly
    # modified
    coords_out = np.copy(coords)
    data_out = np.copy(data)
    
    # Store inputs original shapes
    coords_shape = coords_out.shape
    
    # Convert inputs to numpy arrays
    coords_out = np.asarray(coords_out).squeeze()
    data_out = np.asarray(data_out)
    
    # Check inputs
    if coords_out.ndim != 1:
        raise ValueError('Coordinates are not 1-D!')
        
    if data_out.ndim > 2:
        raise ValueError('Data must be a 2-D matrix!')
    elif data_out.ndim == 1:
        data_out = np.reshape(data_out, (-1, 1))
        
    dim_match = coords_out.size == np.asarray(data_out.shape)
    transpose_flag = False
    if not np.any(dim_match):
        raise IndexError('No apparent agreement')
    elif np.all(dim_match):
        warn(('Ambiguous dimensionalities; assuming columns of data are to ' + 
              'be interpolated'), Warning)
    elif dim_match[0] != 1:
        data_out = data_out.transpose()
        transpose_flag = True
        
    # Determine where NaN coordinates are replace them using linear 
    # interpolation assuming uniform spacing
    uniform_spacing = np.arange(0, coords_out.size)
    coords_nan = np.isnan(coords_out)
    coords_out[coords_nan] = np.interp(uniform_spacing[coords_nan], 
          uniform_spacing[~coords_nan], coords_out[~coords_nan])
    
    # Iterate over each dimension of data
    for ii in range(0, data_out.shape[1]):
        
        # Determine where the NaN data and replace them using linear 
        # interpolation
        data_nan = np.isnan(data_out[:, ii])
        data_out[data_nan, ii] = np.interp(coords_out[data_nan], 
                coords_out[~data_nan], data_out[~data_nan, ii])
        
    # Reshape results to match inputs
    coords_out = np.reshape(coords_out, coords_shape)
    if transpose_flag:
        data_out = np.transpose(data_out)
    
    # Return coordinates and data with NaN values replaced
    return [coords_out, data_out]

def extract_time_stamp():
    array = list()
    time_array = list()
    data = pandas.read_csv(platform_position_data, skiprows=6)
    for elements in data:
        if "Time" in elements:
            array.append(elements)
    for contents2 in array:
        time_array.append(data[contents2][0:].values)
    time_array = np.array(time_array).astype(np.float)
    return time_array

def get_range(radar_xpos, radar_ypos, radar_zpos, img_x, img_y, img_z):
    return math.sqrt((radar_xpos - img_x)**2 + (radar_ypos - img_y)**2 + (radar_zpos - img_z)**2)

radar_data = unpack('UASSAR3_Saturday_pseudoflight_1')
platform_position_data = 'uassar3_saturday_pseudoflight_1.csv'
given_object = 'uassar3 triangle 3.csv'
meters = 3
size = 500
eyeballing_start_time = 160
eyeballing_end_time = 750
time_offset = 12
range_offset = 0.2

time_stamps = radar_data['time_stamp']
scan_data = radar_data['scan_data']
range_bins = radar_data['range_bins']

scan_data = np.array(scan_data).astype(float)
#optional rcs need button for this 
'''
for elements in range(len(scan_data)):
    for elements2 in range(int(len(scan_data[0])*0.75)):
        scan_data[elements][elements2] = scan_data[elements][elements2] * ((range_bins[elements2] - range_offset) **4.0)
'''
plat_pos = extract_platform_position()

#plt.figure()
#plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0],range_bins[-1],(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')

#plt.figure()
#plt.plot(plat_pos)
plat_pos = linear_interp_nan(extract_time_stamp(),plat_pos)[1]
refl_pos = extract_given_object()

range_to_refl = np.sqrt((plat_pos[:,0] - refl_pos[0])**2 + (plat_pos[:,1] - refl_pos[1])**2 + (plat_pos[:,2] - refl_pos[2])**2)
#plt.plot(range_to_refl, np.transpose(extract_time_stamp()))
#^ this is what you plot on top of the unpack image 

plt.figure()
plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0],range_bins[-1],(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
#plt.colorbar()
#plt.clim(60, 90)
plt.plot(range_to_refl, np.transpose(extract_time_stamp()-time_offset))

center = extract_given_object()


#interpolation (aligning radar data and position values)
new_x_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(extract_time_stamp()-time_offset).flatten(),plat_pos[:,0])
new_y_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(extract_time_stamp()-time_offset).flatten(),plat_pos[:,1])
new_z_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(extract_time_stamp()-time_offset).flatten(),plat_pos[:,2])
range_to_refl = np.sqrt((new_x_pos[:] - refl_pos[0])**2 + (new_y_pos[:] - refl_pos[1])**2 + (new_z_pos[:] - refl_pos[2])**2)
#plt.figure()
#plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0]-range_offset,range_bins[-1]-range_offset,(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
#plt.plot(range_to_refl, (time_stamps-time_stamps[0])/1000)

interp_plat_pos = list()
for elements in range(len(new_x_pos)):
    interp_plat_pos.append([new_x_pos[elements],new_z_pos[elements],new_y_pos[elements]])
interp_plat_pos = np.array(interp_plat_pos).squeeze()

x_vec = np.linspace(-meters,meters,size)
y_vec = np.linspace(-meters,meters,size)
scan_data_final = scan_data[eyeballing_start_time:eyeballing_end_time,:]
interp_plat_pos_final = interp_plat_pos[eyeballing_start_time:eyeballing_end_time,:]
x_vec_new = x_vec+center[1]
y_vec_new = y_vec+center[2]
#sar_image = interp_approach(scan_data_final, range_bins-range_offset, interp_plat_pos_final, x_vec_new, y_vec_new)
#plt.figure()
#plt.imshow((np.abs(sar_image)),extent=[x_vec_new[0], x_vec_new[-1], y_vec_new[-1], y_vec_new[0]],aspect = 'auto')