import math
import matplotlib.pyplot as plt
import numpy as np
from pulson440_formats import CONFIG_MSG_FORMAT
from pulson440_constants import SPEED_OF_LIGHT, T_BIN, DN_BIN
import pandas
import timeit
import pickle
import sys

DT_0 = 10
pulse_data = 'railTestDiagonal.pkl'
platform_position_data = 'UASSAR4_rail_diagonal.csv'
given_object = 'triangle.csv'

eyeballing_time_start = 272
eyeballing_end_time = 1400

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
        config = read_config_data(f, legacy)
        
        scan_start_time = float(config['scan_start'])
        scan_end_time = float(config['scan_stop'])
        num_range_bins = DN_BIN * math.ceil((scan_end_time - scan_start_time) /
                                           (T_BIN * 1000 * DN_BIN))
        num_packets_per_scan = math.ceil(num_range_bins / 350)
        start_range = SPEED_OF_LIGHT * ((scan_start_time * 1e-12) - DT_0 * 1e-9) / 2
        drange_bins = SPEED_OF_LIGHT * T_BIN * 1e-9 / 2
        range_bins = start_range + drange_bins * np.arange(0, num_range_bins, 1)
        
        data = dict()
        data= {'scan_data': [],
               'time_stamp': [],
               'packet_ind': [],
               'packet_pulse_ind': [],
               'range_bins': range_bins}
        single_scan_data = []
        packet_count = 0
        pulse_count = 0
        
        while True:
            packet = f.read(1452)
            if len(packet) < 1452:
                break            
            packet_count += 1
            
            data['packet_ind'].append(np.frombuffer(packet[48:50], dtype='u2'))
            
            if packet_count % num_packets_per_scan == 0:
                num_samples = num_range_bins % 350
                packet_data = np.frombuffer(packet[52:(52 + 4 * num_samples)], 
                                                   dtype='>i4')
                single_scan_data.append(packet_data)
                data['scan_data'].append(np.concatenate(single_scan_data))
                data['time_stamp'].append(np.frombuffer(packet[8:12], 
                    dtype='>u4'))
                single_scan_data = []
                pulse_count += 1
            else:
                num_samples = 350
                packet_data = np.frombuffer(packet[52:(52 + 4 * num_samples)], 
                                                   dtype='>i4')
                single_scan_data.append(packet_data)
            
        if single_scan_data:
            single_scan_data = np.concatenate(single_scan_data)
            num_pad = data['scan_data'][0].size - single_scan_data.size
            single_scan_data = np.pad(single_scan_data, (0, num_pad), 
                                      'constant', constant_values=0)
            data['scan_data'].append(single_scan_data)
                
        data['scan_data'] = np.stack(data['scan_data'])
        
        data['time_stamp']

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
    pltpos = extract_platform_position()
    mini = 2 ** 63 -1
    maxi = -2 ** 63 -1
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

def get_end_time_platform():
    pltpos = extract_platform_position()
    mini = 2 ** 63 -1
    maxi = -2 ** 63 -1
    for n in range(1000):
        if pltpos[len(pltpos)-n-1][2] > maxi:
            maxi = pltpos[len(pltpos)-n-1][2] 
        if pltpos[len(pltpos)-n-1][2] < mini:
            mini = pltpos[len(pltpos)-n-1][2]
    end_time = None
    n = len(pltpos) -1
    while end_time == None:
         if abs(pltpos[n][2] - pltpos[n-100][2]) > abs(mini - maxi):
             end_time = n-100
         n -= 1 
    return end_time

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

'''
Plots a picture that illustrates the edges of the source image
@param   arr   the 2D array or list of lists that contains the sum magnitude of pulses
@param   diff  the index that dictates how much contrast the algorithm is looking for
'''
def edge_detection(arr, diff):
    ED = np.zeros((len(arr),len(arr[0]),3))
    for row in range(len(arr)):
        for col in range(len(arr[0])):
           # if col is not len(arr[0]) - 1:
            if (col < len(arr[0]) - 1 and np.absolute(arr[row][col]-arr[row][col+1]) > diff) or (row < len(arr) - 4 and np.absolute(arr[row][col]-arr[row+4][col]) > diff*4):
                ED[row,col,0] = 255
                ED[row,col,1] = 255
                ED[row,col,2] = 255
            else:
                ED[row,col,0] = 1
                ED[row,col,1] = 1
                ED[row,col,2] = 1
    return ED

'''
Displays a grayscaled version of arr which contains the values for the pixels in plt
'''
def gray_scale(arr):
    plt.imshow(arr, cmap='gray')
    
'''
x and y represent pixel coordinate. Difference between start and end x and start and end y
must be less than size, and start x/y must be greater or equal to zero.
def partImage returns a 2d array of magnitudes from a range of pixels defined
by the parameters
NOTE: keep resolution_multiplier value to 1 unless you know what you're doing
'''
def part_image(start_x, start_y, end_x, end_y, pulses, range_bin_d, radar_x, radar_y, radar_z, size, resolution_multiplier):
    resScalar = resolution_multiplier
    xDiff = np.abs(end_x - start_x)*resScalar
    yDiff = np.abs(end_y - start_y)*resScalar
    pulse_arr = list(list(0+0j for ii in np.arange(0, xDiff)) for jj in np.arange(0, yDiff))
    mag_arr = list(list(0+0j for ii in np.arange(0, xDiff)) for jj in np.arange(0, yDiff))
    y = 2.5 - (5.0/size)*start_y
    for ii in np.arange(0, yDiff):
        print(str(ii) + "/" + str(yDiff))
        x = -2.5 + (5.0/size)*start_x
        for jj in np.arange(0, xDiff):
            for kk in np.arange(0, 100):
                distance = get_range(radar_x[kk], radar_y[kk], radar_z[kk], x, y)
                ratio = (distance % range_bin_d) / range_bin_d
                index = math.floor(distance/range_bin_d)
                pulse_arr[ii][jj] += (pulses[kk][index]*(1-ratio) + pulses[kk][index+1]*(ratio))
            mag_arr[ii][jj] = np.abs(pulse_arr[ii][jj])
            x = x + (5.0/(size*resScalar))
        y = y - (5.0/(size*resScalar))
    return mag_arr

'''
returns entropy value of array of pixels in given image
to compare images of different resolutions, you must scale 
the image's entropy appropriately to to the ratio of one 
image's resolution to the other image resolution
'''
def get_entropy(magnitude_array):
    entropy_sum = 0
    minMag = magnitude_array[0][0] 
    maxMag = magnitude_array[0][0]
    for yy in range(len(magnitude_array)):
        for xx in range(len(magnitude_array[yy])):
            curr_mag = magnitude_array[yy][xx]
            #print("max: " + str(maxMag) + "----- min: " + str(minMag))
            if curr_mag > maxMag:
                maxMag = curr_mag
            if curr_mag < minMag:
                minMag = curr_mag
    
    magDiff = np.abs(maxMag - minMag)
    #print("magDiff = " + str(magDiff))
    for yy in range(len(magnitude_array)):
        for xx in range(len(magnitude_array[yy])):
            curr_mag_final = np.abs(magnitude_array[yy][xx] - minMag)/magDiff
            if (curr_mag_final != 0):
                #print("currMagFinal: " + str(curr_mag_final))
                entropy_sum += curr_mag_final*np.log2(curr_mag_final)
    return -1*entropy_sum

def display_menu():
    print("\nThe signals have been successfully processed, please indicate which version of image you would like to view:")
    print("1. Original image")
    print("2. Edge detected")
    print("3. Grayscaled")
    print("4. Quit")
    choice = ""
    while not (choice is "1" or choice is "2" or choice is "3" or choice is "4"):
        choice = input("Choice: ")
    return choice

# Calculating the color of each pixel in the image by iterating over all the pulses and summing the complex values within
# the correct range bins and ultimately storing the absolute value of the sum at the corresponding index in pixel_values 
def linear_interp(pulses,range_bins,radar_x,radar_y,radar_z,size,center,width,height):
    #Initializes grid as a list of lists
    y = height / 2 + center[1]
    pixel_values = np.zeros((size,size))
    range_bin = range_bins[1]-range_bins[0]
    first_range_bin = range_bins[0]
    distance_mat = np.zeros((size, size))
    for ii in range(size):
        x = width * -1 /2 + center[0]
        print(str(ii)+"/"+str(size))
        for jj in range(size):
            for kk in range(len(radar_x)):
                distance = get_range(radar_x[kk], radar_y[kk], radar_z[kk], x, y, 0) - first_range_bin
                distance_mat[ii][jj] = distance
                ratio = (distance % range_bin) / range_bin
                index = math.floor(distance / range_bin)
                pixel_values[ii][jj] += (pulses[kk][index]*(1-ratio) + pulses[kk][index+1]*(ratio))
            pixel_values[ii][jj] = pixel_values[ii][jj]
            x = x + width/size
        y = y - height/size
    return pixel_values
    
def create_SAR_image(pickle_file,size):
    start = timeit.default_timer()
    radar_positions = pickle_file[0]
    pulses = pickle_file[1]
    range_bins = pickle_file[2]
    static_object = [-0.5,1,0] # should be extract_given_object()
    
    #Stores the positions of the UAS
    radar_x = []
    radar_y = []
    radar_z = []
    for position in radar_positions:
        print(position)
        radar_x.append(position[2])
        radar_y.append(position[0])
        radar_z.append(position[1])
    img = np.abs(linear_interp(pulses,range_bins,radar_x,radar_y,radar_z,size,static_object,5.0,4.0))
    
    with open('pixel_values', 'wb') as f:
        pickle.dump(img, f)
        
    choice = ""
    while choice is not "4":
        choice = display_menu()
        if choice is "1":
            plt.imshow(img)
        elif choice is "2":
            ED_val = input("Please input your desired edge detection index 1000-5000 (the higher\nthe index, the more difficult it is to detect edges, but the less noise): ")
            plt.imshow(edge_detection(img,(int)(ED_val)))
        elif choice is "3":
            gray_scale(img)
        if choice is not "4":
            plt.show()
    end = timeit.default_timer()
    print("\nProgram terminated, total run time = " + str(end-start))
    
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
    endplat = get_end_time_platform()
    r = np.array(stamprad).astype(float)
    p = np.array(stamppla).astype(float)
    cutcxpls = abscxpls[strrad:][:]
    cutpltpos = pltpos[strplat:endplat][:]
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
    #print(endrad)
    cxpls = cxpls[strrad:][:]
    if index != None:
        cxpls = cxpls[:index+1][:]
    if len(points) > len(cxpls):
        points = points[:len(points)-1][:]
    cxpls = cxpls[:endrad][:]
    points = points[:endrad][:]
    return [cxpls,points]

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
    print(len(x))
    print(len(final_result))
    plt.figure()
    plt.plot(x)
    plt.figure()
    plt.plot(final_result)
    return final_result

def main(args):
    create_SAR_image(combine_all_arrays(),args)
    #show_image_determine_start_time()

if len(sys.argv) == 1:
    main(60)
else:
    main(sys.argv[1])
###############################################################################################################
def get_parabola():
    cxpls = extract_complex_pulse()
    abscxpls = abs(cxpls)
    distance = list()
    x = list()
    interm = get_start_time_highest_intensity()[1]
    adjust = get_start_time_highest_intensity()[0]
    for n in range(len(abscxpls)):
        distance.append((math.sqrt((interm) ** 2 + (len(abscxpls) / 2 - n) ** 2)))
    adjust_refine = adjust - len(abscxpls) /2
    for n in range(len(abscxpls)):
        x.append(n + adjust_refine)
    return [x,distance]
    
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
    plt.imshow(temp2,extent=[0,10,0,500],aspect = 'auto')
    distancelist = list()
    x = extract_given_object()
    y = np.array(time_align_interpolation()[1])
    for ii in range(y.shape[0]):
        distance = get_range(x[0],x[1],x[2],y[ii][0],y[ii][1],y[ii][2])
        distancelist.append(distance)
    temp = distancelist[0:-1:round(y.shape[0] / temp.shape[0])]
    index = range(len(temp)+eyeballing_time_start,eyeballing_time_start,-1)
    plt.plot(temp, index,'r--')