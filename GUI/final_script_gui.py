from pulson440_unpack import unpack
from backprojection import interp_approach
import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
from warnings import warn
from matplotlib.figure import Figure


class Script:
    def __init__(self, param1, param2, param3):
        self.radar_data = unpack(param1)
        self.platform_position_data = param2
        self.given_object = param3
        self.meters = 0
        self.size = 0
        self.eyeballing_start_time = 0
        self.eyeballing_end_time = 0
        self.time_offset = 0.0
        self.range_offset = 0.0
        

    def extract_platform_position(self):
        array = list()
        new_array = list()
        position_array = list()
        final_result = list()
        data = pandas.read_csv(self.platform_position_data, skiprows=2, low_memory = False)
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
    
    def extract_given_object(self):
        array = list()
        new_array = list()
        position_array = list()
        final_result = list()
        data = pandas.read_csv(self.given_object, skiprows=2, low_memory = False)
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
    
    def linear_interp_nan(self, coords, data):
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
    
    def extract_time_stamp(self):
        array = list()
        time_array = list()
        data = pandas.read_csv(self.platform_position_data, skiprows=6)
        for elements in data:
            if "Time" in elements:
                array.append(elements)
        for contents2 in array:
            time_array.append(data[contents2][0:].values)
        time_array = np.array(time_array).astype(np.float)
        return time_array
    
    def get_range(self, radar_xpos, radar_ypos, radar_zpos, img_x, img_y, img_z):
        return math.sqrt((radar_xpos - img_x)**2 + (radar_ypos - img_y)**2 + (radar_zpos - img_z)**2)
    
    def get_graph(self):
        time_stamps = self.radar_data['time_stamp']
        scan_data = self.radar_data['scan_data']
        range_bins = self.radar_data['range_bins']
        
        scan_data = np.array(scan_data).astype(float)
        #optional rcs need button for this 
        
        for elements in range(len(scan_data)):
            for elements2 in range(int(len(scan_data[0])*0.75)):
                scan_data[elements][elements2] = scan_data[elements][elements2] * ((range_bins[elements2] - self.range_offset) **4.0)
        
        plat_pos = self.extract_platform_position()
        
        #plt.figure()
        #plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0],range_bins[-1],(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
        
        #plt.figure()
        #plt.plot(plat_pos)
        plat_pos = self.linear_interp_nan(self.extract_time_stamp(),plat_pos)[1]
        refl_pos = self.extract_given_object()
        
        range_to_refl = np.sqrt((plat_pos[:,0] - refl_pos[0])**2 + (plat_pos[:,1] - refl_pos[1])**2 + (plat_pos[:,2] - refl_pos[2])**2)
        #plt.plot(range_to_refl, np.transpose(extract_time_stamp()))
        
        #plt.figure()
        #plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0],range_bins[-1],(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
        #plt.colorbar()
        #plt.clim(60, 90)
        #plt.plot(range_to_refl, np.transpose(extract_time_stamp()-time_offset))
        
        center = self.extract_given_object()
        
        #interpolation (aligning radar data and position values)
        new_x_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(self.extract_time_stamp()-self.time_offset).flatten(),plat_pos[:,0])
        new_y_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(self.extract_time_stamp()-self.time_offset).flatten(),plat_pos[:,1])
        new_z_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(self.extract_time_stamp()-self.time_offset).flatten(),plat_pos[:,2])
        range_to_refl = np.sqrt((new_x_pos[:] - refl_pos[0])**2 + (new_y_pos[:] - refl_pos[1])**2 + (new_z_pos[:] - refl_pos[2])**2)
        #plt.figure()
        #plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0]-self.range_offset,range_bins[-1]-self.range_offset,(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
        return (range_to_refl, (time_stamps-time_stamps[0])/1000)
    def get_image_array(self, time_shift, size):
        time_offset =  self.get_time_offset() + time_shift
    
        time_stamps = self.radar_data['time_stamp']
        scan_data = self.radar_data['scan_data']
        range_bins = self.radar_data['range_bins']
    
        scan_data = np.array(scan_data).astype(float)
        #optional rcs need button for this 
        '''
        for elements in range(len(scan_data)):
            for elements2 in range(int(len(scan_data[0])*0.75)):
                scan_data[elements][elements2] = scan_data[elements][elements2] * ((range_bins[elements2] - range_offset) **4.0)
        '''
        plat_pos = self.extract_platform_position()
    
        #plt.figure()
        #plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0],range_bins[-1],(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
    
        #plt.figure()
        #plt.plot(plat_pos)
        plat_pos = self.linear_interp_nan(self.extract_time_stamp(),plat_pos)[1]
        refl_pos = self.extract_given_object()
    
        range_to_refl = np.sqrt((plat_pos[:,0] - refl_pos[0])**2 + (plat_pos[:,1] - refl_pos[1])**2 + (plat_pos[:,2] - refl_pos[2])**2)
        #plt.plot(range_to_refl, np.transpose(extract_time_stamp()))
    
        #plt.figure()
        #plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0],range_bins[-1],(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
        #plt.colorbar()
        #plt.clim(60, 90)
        #plt.plot(range_to_refl, np.transpose(extract_time_stamp()-time_offset))
    
        center = self.extract_given_object()
    
    
        #interpolation (aligning radar data and position values)
        new_x_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(self.extract_time_stamp()-time_offset).flatten(),plat_pos[:,0])
        new_y_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(self.extract_time_stamp()-time_offset).flatten(),plat_pos[:,1])
        new_z_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(self.extract_time_stamp()-time_offset).flatten(),plat_pos[:,2])
        range_to_refl = np.sqrt((new_x_pos[:] - refl_pos[0])**2 + (new_y_pos[:] - refl_pos[1])**2 + (new_z_pos[:] - refl_pos[2])**2)
        #plt.figure()
        #plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0]-range_offset,range_bins[-1]-range_offset,(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
        #plt.plot(range_to_refl, (time_stamps-time_stamps[0])/1000)
    
        interp_plat_pos = list()
        for elements in range(len(new_x_pos)):
            interp_plat_pos.append([new_x_pos[elements],new_z_pos[elements],new_y_pos[elements]])
        interp_plat_pos = np.array(interp_plat_pos).squeeze()
    
        x_vec = np.linspace(-self.meters,self.meters,size)
        y_vec = np.linspace(-self.meters,self.meters,size)
        scan_data_final = scan_data[self.eyeballing_start_time:self.eyeballing_end_time,:]
        interp_plat_pos_final = interp_plat_pos[self.eyeballing_start_time:self.eyeballing_end_time,:]
        x_vec_new = x_vec+center[1]
        y_vec_new = y_vec+center[2]
        sar_image = interp_approach(scan_data_final, range_bins-self.range_offset, interp_plat_pos_final, x_vec_new, y_vec_new)
        #plt.figure()
        #plt.imshow((np.abs(sar_image)),extent=[x_vec_new[0], x_vec_new[-1], y_vec_new[-1], y_vec_new[0]],aspect = 'auto')
        return sar_image
    
    def get_time_offset(self):
        return self.time_offset
    
    def get_entropy(self, magnitude_array):
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
    
    def testEntropy(self, deviation, step, resolution):
        entropyArr = list()
        combineArr = self.get_image_array(0, resolution)
        entropyArr.append(self.get_entropy(np.abs(combineArr)))
        counter = 0
        for ii in np.arange(-deviation, deviation, step):
            print(str(counter) + "/" + str(len(np.arange(-deviation, deviation, step))))
            combineArr = self.get_image_array(ii, resolution)
            entropyArr.append(self.get_entropy(np.abs(combineArr)))
            counter = counter + 1
        
        bestIndex = 0
        bestEntropy = entropyArr[0]
        for ii in range(len(entropyArr)):
            if entropyArr[ii] < bestEntropy:
                bestIndex = ii
                bestEntropy = entropyArr[ii]
                print("(index, entropy): (" + str(ii) + ", " + str(entropyArr[ii]) + ")")
            
        print("BEST (index, entropy): (" + str(bestIndex) + ", " + str(bestEntropy) + ")")
        if bestIndex != 0:
            print("BEST TIME SHIFT VALUE: " + str(-deviation+(bestIndex*step)))
        else:
            print("BEST TIME SHIFT VALUE: 0")
        
        finalShift = 0
        if bestIndex != 0:
            finalShift = -deviation+(bestIndex*step)
        
        return finalShift
    
    def main_entropy(self, param4, param5, param6, param7, param8, param9):
        self.meters = param4
        self.size = param5
        self.eyeballing_start_time = param6
        self.eyeballing_end_time = param7
        self.time_offset = param8
        self.range_offset = param9
        
        shift_test = self.testEntropy(.5, .05, 100)
        
        f = Figure(figsize=(5, 5), dpi=100)
        ax = f.add_subplot(111)
        img = ax.imshow(np.absolute(self.get_image_array(shift_test, 500)))
        return (f, ax, img)
    
        
    def main_func(self, param4, param5, param6, param7, param8, param9):
        
        self.meters = param4
        self.size = param5
        self.eyeballing_start_time = param6
        self.eyeballing_end_time = param7
        self.time_offset = param8
        self.range_offset = param9
        
        time_stamps = self.radar_data['time_stamp']
        scan_data = self.radar_data['scan_data']
        range_bins = self.radar_data['range_bins']
        
        scan_data = np.array(scan_data).astype(float)
        #optional rcs need button for this 
        '''
        for elements in range(len(scan_data)):
            for elements2 in range(int(len(scan_data[0])*0.75)):
                scan_data[elements][elements2] = scan_data[elements][elements2] * ((range_bins[elements2] - self.range_offset) **4.0)
        '''
        plat_pos = self.extract_platform_position()
        
        #plt.figure()
        #plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0],range_bins[-1],(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
        
        #plt.figure()
        #plt.plot(plat_pos)
        plat_pos = self.linear_interp_nan(self.extract_time_stamp(),plat_pos)[1]
        refl_pos = self.extract_given_object()
        
        range_to_refl = np.sqrt((plat_pos[:,0] - refl_pos[0])**2 + (plat_pos[:,1] - refl_pos[1])**2 + (plat_pos[:,2] - refl_pos[2])**2)
        #plt.plot(range_to_refl, np.transpose(extract_time_stamp()))
        
        #plt.figure()
        #plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0],range_bins[-1],(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
        #plt.colorbar()
        #plt.clim(60, 90)
        #plt.plot(range_to_refl, np.transpose(extract_time_stamp()-time_offset))
        
        center = self.extract_given_object()
        print(center)
        
        #interpolation (aligning radar data and position values)
        new_x_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(self.extract_time_stamp()-self.time_offset).flatten(),plat_pos[:,0])
        new_y_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(self.extract_time_stamp()-self.time_offset).flatten(),plat_pos[:,1])
        new_z_pos = np.interp((time_stamps-time_stamps[0])/1000,np.transpose(self.extract_time_stamp()-self.time_offset).flatten(),plat_pos[:,2])
        range_to_refl = np.sqrt((new_x_pos[:] - refl_pos[0])**2 + (new_y_pos[:] - refl_pos[1])**2 + (new_z_pos[:] - refl_pos[2])**2)
        #plt.figure()
        #plt.imshow(20 * np.log10(np.abs(scan_data)),extent=[range_bins[0]-range_offset,range_bins[-1]-range_offset,(time_stamps[-1]-time_stamps[0])/1000,0],aspect = 'auto')
        #plt.plot(range_to_refl, (time_stamps-time_stamps[0])/1000)
        
        interp_plat_pos = list()
        for elements in range(len(new_x_pos)):
            interp_plat_pos.append([new_x_pos[elements],new_z_pos[elements],new_y_pos[elements]])
        interp_plat_pos = np.array(interp_plat_pos).squeeze()
        
        x_vec = np.linspace(-self.meters,self.meters,self.size)
        y_vec = np.linspace(-self.meters,self.meters,self.size)
        scan_data_final = scan_data[self.eyeballing_start_time:self.eyeballing_end_time,:]
        interp_plat_pos_final = interp_plat_pos[self.eyeballing_start_time:self.eyeballing_end_time,:]
        x_vec_new = x_vec+center[1]
        y_vec_new = y_vec+center[2]
        sar_image = interp_approach(scan_data_final, range_bins-self.range_offset, interp_plat_pos_final, x_vec_new, y_vec_new)
        f = Figure(figsize=(5, 5), dpi=100)
        ax = f.add_subplot(111)
        img = ax.imshow(20*np.log10((np.abs(sar_image))),extent=[x_vec_new[0], x_vec_new[-1], y_vec_new[-1], y_vec_new[0]],aspect = 'auto')
        return (f, ax, img)