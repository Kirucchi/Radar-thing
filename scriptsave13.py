import math
import matplotlib.pyplot as plt
import numpy as np
from pulson440_formats import CONFIG_MSG_FORMAT
from pulson440_constants import SPEED_OF_LIGHT, T_BIN, DN_BIN
import pandas
import timeit
import pickle
from warnings import warn

DT_0 = 10
pulse_data = 'railTestDiagonal.pkl'
platform_position_data = 'UASSAR4_rail_diagonal.csv'
given_object = 'triangle.csv'

size = 50
eyeballing_time_start = 272
eyeballing_end_time = 1400
meters = 4

def read_config_data(file_handle, legacy=False):
    """
    Read in configuration data based on platform.
    """
    config = dict.fromkeys(CONFIG_MSG_FORMAT.keys())
    
    if legacy:
        config_msg = file_handle.read(44)
        config['node_id'] = np.frombuffer(config_msg[4:8], dtype='>u4')[0]
        config['scan_start'] = np.frombuffer(config_msg[8:12], dtype='>i4')[0]
        config['scan_stop'] = np.frombuffer(config_msg[12:16], dtype='>i4')[0]
        config['scan_res'] = np.frombuffer(config_msg[16:18], dtype='>u2')[0]
        config['pii'] = np.frombuffer(config_msg[18:20], dtype='>u2')[0]
        config['ant_mode'] = np.uint16(config_msg[32])
        config['tx_gain_ind'] = np.uint16(config_msg[33])
        config['code_channel'] = np.uint16(config_msg[34])
        config['persist_flag'] = np.uint16(config_msg[35])
        
    else:
        config_msg = file_handle.read(32)
        byte_counter = 0
        for config_field in CONFIG_MSG_FORMAT.keys():
            num_bytes = CONFIG_MSG_FORMAT[config_field].itemsize
            config_data = config_msg[byte_counter:(byte_counter + num_bytes)]
            config[config_field] = np.frombuffer(config_data,
                  dtype=CONFIG_MSG_FORMAT[config_field])[0]
            config[config_field] = config[config_field].byteswap()
            byte_counter += num_bytes
            
    return config

def unpack(file, legacy=False):
    """
    Unpacks PulsOn 440 radar data from input file
    """
    with open(file, 'rb') as f:
        
        # Read configuration part of data
        config = read_config_data(f, legacy)
        
        # Compute range bins in datas
        scan_start_time = float(config['scan_start'])
        start_range = SPEED_OF_LIGHT * ((scan_start_time * 1e-12) - DT_0 * 1e-9) / 2
        
        # Read data
        data = dict()
        data= {'scan_data': [],
               'time_stamp': [],
               'packet_ind': [],
               'packet_pulse_ind': [],
               'range_bins': [],
               'config': config}
        single_scan_data = []
        packet_count = 0
        pulse_count = 0
        
        while True:
            
            # Read a single data packet and break loop if not a complete packet
            # (in terms of size)
            packet = f.read(1452)
            if len(packet) < 1452:
                break            
            
            # Get information from first packet about how scans are stored and 
            # range bins collected
            if packet_count == 0:
                num_range_bins = np.frombuffer(packet[44:48], dtype='>u4')[0]
                num_packets_per_scan = np.frombuffer(packet[50:52], dtype='>u2')[0]
                drange_bins = SPEED_OF_LIGHT * T_BIN * 1e-9 / 2
                range_bins = start_range + drange_bins * np.arange(0, num_range_bins, 1)
            packet_count += 1
            
            # Number of samples in current packet and packet index
            num_samples = np.frombuffer(packet[42:44], dtype='>u2')[0]
            data['packet_ind'].append(np.frombuffer(packet[48:50], dtype='>u2')[0])
            
            # Extract radar data samples from current packet; process last 
            # packet within a scan seperately to get all data
            packet_data = np.frombuffer(packet[52:(52 + 4 * num_samples)], 
                                               dtype='>i4')
            single_scan_data.append(packet_data)
            
            if packet_count % num_packets_per_scan == 0:
                data['scan_data'].append(np.concatenate(single_scan_data))
                data['time_stamp'].append(np.frombuffer(packet[8:12], 
                    dtype='>u4')[0])
                single_scan_data = []
                pulse_count += 1
            
        # Add last partial scan if present
        if single_scan_data:
            single_scan_data = np.concatenate(single_scan_data)
            num_pad = data['scan_data'][0].size - single_scan_data.size
            single_scan_data = np.pad(single_scan_data, (0, num_pad), 
                                      'constant', constant_values=0)
            data['scan_data'].append(single_scan_data)
                
        # Stack scan data into 2-D array 
        # (rows -> pulses, columns -> range bins)
        data['scan_data'] = np.stack(data['scan_data'])
        
        # Finalize entries in data
        data['time_stamp'] = np.asarray(data['time_stamp'])
        data['range_bins'] = range_bins

        return data

def extract_complex_pulse():
    #data = unpack(pulse_data)
    f = open('railTestDiagonal.pkl', 'rb')
    data = pickle.load(f)
    f.close()
    
    return data['scan_data']

def extract_time_stamp():
    #data = unpack(pulse_data)
    f = open('railTestDiagonal.pkl', 'rb')
    data = pickle.load(f)
    f.close()
    
    return data['time_stamp']

def extract_range_bins():
    #data = unpack(pulse_data)
    f = open('railTestDiagonal.pkl', 'rb')
    data = pickle.load(f)
    f.close()
    
    return data['range_bins'] 

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

def extract_time_stamp2():
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

def get_start_time_platform():
    '''
    pltpos = extract_platform_position()
    mini = 2 ** 63 -1
    maxi = -2 ** 63 +1
    for n in range(1000):
        if pltpos[n][2] > maxi:
            maxi = pltpos[n][2] 
        if pltpos[n][2] < mini:
            mini = pltpos[n][2]
    start_time = None
    n = 0
    while start_time == None:
         if abs(pltpos[n+100][2] - pltpos[n][2]) > abs(mini - maxi):
             start_time = n+100
         n += 1 
    return start_time
    '''
    pltpos = extract_platform_position()
    plat_var = np.mean(np.diff(pltpos[0:1000, 2]))
    start_time = None
    n = 0
    while start_time == None:
         if abs(pltpos[n, 2] - pltpos[n+1, 2]) **2 > 5*plat_var:
             start_time = n
         n += 1 
    return start_time

def get_range(radar_xpos, radar_ypos, radar_zpos, img_x, img_y, img_z):
    return math.sqrt((radar_xpos - img_x)**2 + (radar_ypos - img_y)**2 + (radar_zpos - img_z)**2)

def combine_all_arrays():
    pickle_file = list()
    x = time_align_interpolation()
    pickle_file.append(x[1])
    pickle_file.append(x[0])
    pickle_file.append(extract_range_bins())
    pickle_file = np.array(pickle_file)
    return pickle_file

def create_SAR_image(pickle_file,size):
    start = timeit.default_timer()

    radar_positions = pickle_file[0]
    pulses = pickle_file[1]
    range_bins = pickle_file[2]
    static_object = extract_given_object()
    
    radar_x = []
    for position in radar_positions:
        radar_x.append(position[2])
    radar_y = []
    for position in radar_positions:
        radar_y.append(position[0])
    radar_z = []
    for position in radar_positions:
        radar_z.append(position[1])
    center = [0,0,0]
    
    for elements in range(len(radar_x)):
        if np.isnan(radar_x[elements]):
            n = 1
            while np.isnan(radar_x[elements+n]) or np.isnan(radar_x[elements-n]):
                n += 1
            if np.isnan(radar_x[elements]):
                radar_x[elements] = (radar_x[elements-n] + radar_x[elements+n])/2
            if np.isnan(radar_y[elements]):
                radar_y[elements] = (radar_y[elements-n] + radar_y[elements+n])/2
            if np.isnan(radar_z[elements]):
                radar_z[elements] = (radar_z[elements-n] + radar_z[elements+n])/2
    
    y = meters+center[2]
    temp = meters
    pixel_values = np.zeros((size,size))
    range_bin = range_bins[1]-range_bins[0]
    first_range_bin = range_bins[0]
    distance_mat = np.zeros((size, size))
    
    for ii in range(size):
        x = -meters+center[1]
        print(str(ii)+"/"+str(size))
        for jj in range(size):
            for kk in range(0,500):#range(int(len(radar_x)/2)-1,int(len(radar_x)/2)):
                distance = get_range(radar_x[kk], radar_y[kk], radar_z[kk], x, y, 0) - first_range_bin
                distance_mat[ii][jj] = distance
                ratio = (distance % range_bin) / range_bin                
                index = math.floor(distance / range_bin)
                pixel_values[ii][jj] += (pulses[kk][index]*(1-ratio) + pulses[kk][index+1]*(ratio))
            pixel_values[ii][jj] = pixel_values[ii][jj]
            x = x + temp/size
        y = y - temp/size
    
    plt.figure()
    plt.imshow(np.abs(pixel_values))

    with open('pixel_values', 'wb') as f:
        pickle.dump(pixel_values, f)
        
    end = timeit.default_timer()
    print(end-start)
    
def intersect():
    cxpls = extract_complex_pulse()
    abscxpls = abs(cxpls)
    plt.imshow(abscxpls)
    #x = get_parabola()[0]
    #distance = get_parabola()[1]
    #plt.plot(distance,x,'r--')    
    
def show_image_determine_start_time():
    intersect()
    
def time_align_interpolation():
    cxpls = extract_complex_pulse()
    abscxpls = abs(cxpls)
    pltpos = extract_platform_position()
    strplat = get_start_time_platform()
    strrad = eyeballing_time_start
    stamprad = list(map(float,extract_time_stamp()))
    stamppla = extract_time_stamp2()
    pltpos = linear_interp_nan(stamppla, pltpos)[1]
    stamppla = linear_interp_nan(stamppla, pltpos)[0]
    r = np.array(stamprad).astype(float)
    p = np.array(stamppla).astype(float)
    cutcxpls = abscxpls[strrad:][:]
    cutpltpos = pltpos[strplat:][:]
    freqr = 1/(float(r[1]) - float(r[0])) * 1000
    freqp = 1/(p[0][1] - p[0][0]) 
    divide = float(freqp / freqr)
    points = list()
    points.append(pltpos[strplat])
    n = 0
    index = None
    for elements in range(len(cutcxpls)):
        floor = math.floor(divide * elements)
        if floor + 1 < len(cutpltpos):
            decimal = divide * elements - floor
            points.append((cutpltpos[floor+1] - cutpltpos[floor])*decimal + cutpltpos[floor])
        else:
            if index == None:
                index = n
        n += 1
    endrad = eyeballing_end_time - strrad
    cxpls = cxpls[strrad:][:]
    if index != None:
        cxpls = cxpls[:index+1][:]
    if len(points) > len(cxpls):
        points = points[:len(points)-1][:]
    cxpls = cxpls[:endrad][:]
    points = points[:endrad][:]
    x = extract_given_object()
    distance = list()
    axis = list()
    print(x)
    print(points[1])
    print((x[0] - points[1][0])**2)
    print((x[1] - points[1][1])**2)
    print((x[2] - points[1][2])**2)
    for elements in range(600):
        axis.append(elements)
        distance.append(get_range(x[0],x[1],x[2],points[elements][0],points[elements][1],points[elements][2]) +3)
    plt.figure()
    plt.plot(distance, axis,'r-')
    temp = 20*np.log10(np.abs(cxpls))
    max_val = np.max(temp.flatten())
    plt.imshow(temp,extent = [3,10,600,0],aspect = 'auto')
    plt.colorbar()
    plt.clim(max_val - 30, max_val)
    plt.figure()
    plt.plot(points)
    return [cxpls,points]
    
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

def kiryumain():
    create_SAR_image(combine_all_arrays(),size)

time_align_interpolation()
###############################################################################################################
def test():
    final_result = list()
    lister = list()
    data = pandas.read_csv('blah.csv', low_memory = False)
    for contents2 in data:
        lister.append(data[contents2][:].values)
    for contents3 in range(len(lister[0])):
        mini_array = list()
        for contents4 in range(len(lister)):
            mini_array.append(lister[contents4][contents3])
        final_result.append(np.array(mini_array))
    final_result = np.matrix(final_result)
    final_result = np.transpose(final_result)
    x = time_align_interpolation()[1]
    return final_result

def get_parabola():
    cxpls = extract_complex_pulse()
    abscxpls = abs(cxpls)
    print(len(cxpls))
    pltpos = extract_platform_position()
    print(len(pltpos))
    point = extract_given_object()
    distance = list()
    x = list()
    for n in range(len(abscxpls)):
        distance.append(get_range(point[0],point[1],point[2],pltpos[n][0],pltpos[n][1],pltpos[n][2]))
    for n in range(len(abscxpls)):
        x.append(n)
    plt.plot(distance,x)
    
def get_start_time_pulses(): 
    cxpls = extract_complex_pulse()
    abscxpls = abs(cxpls)
    mini = 2 ** 63 -1
    maxi = -2 ** 63 -1
    for n in range(40):
        if abscxpls[n][0] > maxi:
            maxi = abscxpls[n][0] 
        if abscxpls[n][0] < mini:
            mini = abscxpls[n][0]
    interm = None
    n = 0
    while interm == None:
         if abs(abscxpls[n+5][2] - abscxpls[n][2]) > abs(mini - maxi):
             interm = n+5
         n += 1
    return interm

def get_start_time_highest_intensity(): 
    cxpls = extract_complex_pulse()
    abscxpls = abs(cxpls)
    maxi = -2 ** 63 -1
    for n in range(len(abscxpls[0])):
        for no in range(len(abscxpls)):
            if abscxpls[no][n] > maxi:
                maxi  = abscxpls[no][n]
    for n in range(len(abscxpls[0])):
        for no in range(len(abscxpls)):
            if abscxpls[no][n] == maxi:
                return [no,n]
    
def average_5_pixels(x,y,abscxpls): 
    return (abscxpls[x-2][y] + abscxpls[x-1][y] + abscxpls[x][y] + abscxpls[x+1][y] + abscxpls[x+2][y]) / 5.0

def debug():
    temp = abs(extract_complex_pulse())
    strrad = eyeballing_time_start
    temp = abs(extract_complex_pulse())
    temp2 = temp[strrad:(strrad+500)][:]
    #plt.imshow(temp2,extent=[0,10,0,500],aspect = 'auto')
    distancelist = list()
    x = extract_given_object()
    y = np.array(time_align_interpolation()[1])
    for ii in range(y.shape[0]):
        distance = get_range(x[0],x[1],x[2],y[ii][0],y[ii][1],y[ii][2])
        distancelist.append(distance)
    temp = distancelist[0:-1:round(y.shape[0] / temp.shape[0])]
    index = range(len(temp)+eyeballing_time_start,eyeballing_time_start,-1)
    #plt.plot(temp, index,'r--')