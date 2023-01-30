import numpy as np
import scipy
from scipy.integrate import odeint

# audio stuff
from scipy.fftpack import fft
from scipy.fftpack import ifft
# import soundfile as sf
from scipy.signal import spectrogram
import matplotlib.pyplot as plt



"""
Each class in this script is a "Current" object.
"""

class multiply_multi_current_object():
    def __init__(self, current_objects_list, set_max = None):
        """
        Args:
            current_objects_list (list): list of current objects
            set_max (float): if not set to None, then any function calculated here will have max absolute value set_max.
        """
        self.current_objects_list = current_objects_list
        self.name = ''
        for a_current_object in current_objects_list:
            self.name += a_current_object.name +','
        self.extra_descriptors = ''
        for a_current_object in current_objects_list:
            self.extra_descriptors += a_current_object.extra_descriptors
        self.set_max = set_max

    def function(self, N, t):
        combined_I_ext = np.ones((N))
        for a_current_object in self.current_objects_list:
            combined_I_ext = np.multiply(combined_I_ext, a_current_object.function(N, t))
            # print(a_current_object.name)
            # if a_current_object.name == 'I_flat_cutoff_reverse':
            #     print(combined_I_ext)
        if self.set_max != None:
            max_abs_value = np.amax(np.fabs(combined_I_ext))
            if max_abs_value!=0:
                combined_I_ext = np.multiply(self.set_max, np.divide(combined_I_ext, max_abs_value))
        return combined_I_ext

class sum_multi_current_object():
    def __init__(self, current_objects_list, set_max = None):
        """
        Args:
            current_objects_list (list): list of current objects
            set_max (float): if not set to None, then any function calculated here will have max absolute value set_max.
        """
        self.current_objects_list = current_objects_list
        self.name = ''
        for a_current_object in current_objects_list:
            self.name += a_current_object.name +','
        self.extra_descriptors = ''
        for a_current_object in current_objects_list:
            self.extra_descriptors += a_current_object.extra_descriptors
        self.set_max = set_max

    def function(self, N, t):
        combined_I_ext = np.zeros((N))
        for a_current_object in self.current_objects_list:
            combined_I_ext += a_current_object.function(N, t)
        if self.set_max != None:
            max_abs_value = np.amax(np.abs(combined_I_ext))
            combined_I_ext = np.multiply(self.set_max, np.divide(combined_I_ext, max_abs_value))
        return combined_I_ext

class append_multi_current_object():
    def __init__(self, current_objects_list, current_objs_dims_list):
        """
        From two stimuli with N_x and N_y dimensions separately, create a larger stimulus with N_x + N_y dimensions
        The elements of each of the two input argument lists correspond to each other.
        Args:
            current_objects_list (list): list of current objects
        """
        self.current_objects_list = current_objects_list
        self.name = ''
        for a_current_object in current_objects_list:
            self.name += a_current_object.name +','
        self.extra_descriptors = ''
        for a_current_object in current_objects_list:
            self.extra_descriptors += a_current_object.extra_descriptors
        self.extra_descriptors += '_appended'
        self.current_objs_dims_list = current_objs_dims_list

    def function(self, N, t):
        combined_I_ext = np.zeros((N))
        dims_used = 0
        # check to make sure the sum of all current object dimensions = N before continuing
        assert np.sum(np.array(self.current_objs_dims_list)) == N

        for ind, a_current_object in enumerate(self.current_objects_list):
            dims = self.current_objs_dims_list
            combined_I_ext[dims_used : dims_used + dims[ind]] += np.array(a_current_object.function(self.current_objs_dims_list[ind], t))
            dims_used += dims[ind]
        return combined_I_ext

    def prepare_f(self, times_array):
        for ind, a_current_object in enumerate(self.current_objects_list):
            if getattr(a_current_object, 'prepare_f') and callable(getattr(a_current_object, 'prepare_f')):
                a_current_object.prepare_f(times_array)


class freeze_time_current_object():
    def __init__(self, current_object, boundary_pair):
        """
        Args:
            current_objects:
            boundary_pair: expects (time_start_freeze, time_end_freeze)
        """
        self.boundary_pair = boundary_pair
        self.name = ''
        self.name += current_object.name +',freeze('+str(self.boundary_pair)+'),'
        self.current_object = current_object
        self.extra_descriptors = ''
        self.extra_descriptors += current_object.extra_descriptors

    def prepare_f(self, args):
        """
        Used to prepare function in cases like L63 object if the function exists inside that object
        """
        self.current_object.prepare_f(args)

    def function(self, N, t):
        time_initial_of_freeze = self.boundary_pair[0]
        time_final_of_freeze = self.boundary_pair[1]
        if t<time_initial_of_freeze:
            return self.current_object.function(N,t)
        if t>=time_initial_of_freeze and t<time_final_of_freeze:
            return self.current_object.function(N,time_initial_of_freeze)
        if t>=time_final_of_freeze:
            return self.current_object.function(N,t-(time_final_of_freeze-time_initial_of_freeze))

# Currents
class I_flat():
    def __init__(self, magnitude = 30):
        """
        Args:
            magnitude (float): magnitude of current supplied to all neurons at all times
        """
        self.name = "I_flat"
        self.magnitude = magnitude
        self.extra_descriptors = ('magnitude='+str(magnitude)).replace('.','p')

    def function(self,N,t):
        I_ext = self.magnitude*np.ones((N))
        return I_ext

class I_flat_random_targets():
    """
    Stimulates random neurons with density given by argument
    """
    def __init__(self, N, target_array = None, magnitude = 30, density = 0.01):
        self.name = "I_flat_random_targets"
        self.density = density
        # Creating target_array
        self.target_array = target_array
        if target_array == None:
            self.target_array = np.random.rand(N)
            # element is 1 if targeted, 0 if not
            self.target_array[self.target_array > self.density] = 0
            self.target_array[self.target_array > 0] = 1
        self.magnitude = magnitude
        self.extra_descriptors = ('magnitude='+str(magnitude)).replace('.','p')

    def function(self,N,t):
        I_ext = self.magnitude*self.target_array
        # for i, an_el in enumerate(I_ext):
        #         if(an_el!=0):
        #             print((i))
        # print(I_ext[:30])
        # print(np.sum(I_ext))
        return I_ext


class I_flat_random_noise():
    def __init__(self, magnitude = 30, density = 0.01):
        self.name = "I_flat_random_noise"
        self.density = density
        self.magnitude = magnitude
        self.extra_descriptors = ('magnitude='+str(magnitude)).replace('.','p')

    def function(self,N,t):
        I_ext = self.magnitude*np.ones((N))
        random_array = np.random.rand(N)
        random_array -=0.5
        I_ext = np.multiply(I_ext, random_array)
        return I_ext


class I_flat_cutoff():

    def __init__(self, cutoff_time, magnitude = 1):
        self.name = "I_flat_cutoff"
        self.magnitude = magnitude
        self.cutoff_time = cutoff_time
        self.extra_descriptors = ('cutoff='+str(cutoff_time)).replace('.','p')

    def function(self, N,t):
        """
        :param N: Included only because many other functions of current objects need N, and this is standardized
        in code that uses current objects' functions.
        :param t: time (scalar)
        :return: current vector
        """
        # Shortened current meant to drive neurons for a small amount of time
        # and cause them to hopefully complete the signal
        I_ext = self.magnitude*np.ones((N))
        if t>self.cutoff_time:
            I_ext = np.zeros((N))
        return I_ext


class I_flat_cutoff_reverse():

    def __init__(self, cutoff_time, magnitude = 1):
        self.name = "I_flat_cutoff_reverse"
        self.magnitude = magnitude
        self.cutoff_time = cutoff_time
        self.extra_descriptors = ('cutoff='+str(cutoff_time)).replace('.','p')

    def function(self, N,t):
        """
        :param N: Included only because many other functions of current objects need N, and this is standardized
        in code that uses current objects' functions.
        :param t: time (scalar)
        :return: current vector
        """
        # Shortened current meant to play only later parts of signal, not earlier parts
        I_ext = self.magnitude*np.ones((N))
        if t<self.cutoff_time:
            I_ext = np.zeros((N))
        return I_ext

class I_select_spatial_components():

    def __init__(self, num_dims, chosen_dims=[1], I_max=1):
        self.name = "I_flat_3_and_5_only"
        self.num_dims = num_dims
        self.chosen_dims = chosen_dims
        self.I_max = I_max
        self.extra_descriptors = ('spatial_dims'+str(self.chosen_dims))

    def function(self, N, t):
        I_ext = self.I_max * np.zeros((self.num_dims))

        # Make sure the correct neurons (3 and 5) receive stimulus
        for i in self.chosen_dims:
            I_ext[i] = 1
        return I_ext

class I_flat_3_and_5_only():

    def __init__(self):
        self.name = "I_flat_3_and_5_only"

    def function(self, N,t,I_max=30):
        I_ext = I_max * np.ones((N))
        if t<40:
            I_ext = I_max*np.ones((N))
        else:
            I_ext = np.zeros((N))

        # Make sure the correct neurons (3 and 5) receive stimulus
        for i in range(N):
            if i!=3 and i!=5:
                I_ext[i] = -1
        return I_ext

class I_sine():

    def __init__(self, magnitude=30, frequency=0.01, cut_time = None):
        self.name = "I_sine"
        self.magnitude = magnitude
        self.frequency = frequency
        self.omega = frequency*(2*np.pi)
        # extra descriptors for plot titles and file names. Must replace . with p to save files safely:
        self.extra_descriptors = ('I_max='+str(magnitude)+';'+'f='+str(frequency)).replace('.','p')
        self.cut_time = cut_time # time beyond which to cut the signal completely

    def function(self, N, t):
        I_ext = self.magnitude*np.cos(self.omega*t)*np.ones((N))
        if (self.cut_time is not None) and (t >= self.cut_time):
            I_ext = np.zeros((N))
        return I_ext

class I_flat_alternating_steps():
    def __init__(self, magnitude=30, I_dt=100, steps_height_list = [5,5,5,5,0]):
        """
        Args:
            magnitude (float): magnitude of current supplied to all neurons at all times
        """
        self.name = "I_flat_alternating_steps"
        self.magnitude = magnitude
        self.extra_descriptors = ('magnitude='+str(magnitude)).replace('.','p')
        self.steps_height_list = steps_height_list
        self.I_dt = I_dt # time between alternating step heights

    def function(self,N,t):
        steps_height_list = self.steps_height_list
        I_ext = self.magnitude * np.ones((N))
        if t<self.I_dt:
            I_ext *= self.steps_height_list[0]
        for i in range(1,len(steps_height_list)):
            if i*self.I_dt < t and t < (i+1)*self.I_dt:
                I_ext *= self.steps_height_list[i]
        return I_ext



class L63_object():
    def __init__(self, rho=28.0, sigma=10.0, beta=8.0 / 3.0, noise=0, scaling_time_factor=1.0):
        self.name = "I_L63"
        self.extra_descriptors = ('rho=' + str(rho) +",sig="+str(sigma)+",beta="+str(beta))
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.state0 = [-3.1, -3.1, 20.7]
        self.noise = noise
        self.stored_state = np.array([])
        self.interp_function = None
        self.scaling_time_factor = scaling_time_factor

    def dfdt(self, state, t):
        """To be used in odeint"""
        # runs forward in time from 0, cannot just compute at arbitrary t
        x, y, z = state  # Unpack the state vector
        added_noise = self.noise*scipy.random.uniform(low=0, high=1, size=3)
        dstatedt = (1+added_noise)*(self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z)  # Derivatives
        return np.multiply(self.scaling_time_factor,dstatedt)

    def function(self,N,t):
        return self.interp_function(t)

    def prepare_f(self, times_array):
        """
        This needs to be run before the function 'function' can be used for this object.
        :param times_array: array of times at which to produce solution
        :return: states vector from run at times in times_array
        """
        t = times_array

        states = odeint(self.dfdt, self.state0, t)
        self.stored_state = states
        self.interp_function = scipy.interpolate.interp1d(times_array, self.stored_state.transpose(), kind='cubic')

        return states

class L96_object():
    def __init__(self, dims=6, noise=0, scaling_time_factor=1.0):
        self.name = "I_L96"
        self.F = 8
        self.extra_descriptors = ('F=' + str(self.F))
        self.noise = noise
        self.stored_state = np.array([])
        self.interp_function = None
        self.N = dims
        self.state0 = self.F * np.ones(self.N)  # Initial state (equilibrium)
        self.state0[2] += 0.01  # Add small perturbation to 20th variable, default index 19
        self.scaling_time_factor = scaling_time_factor

    def dfdt(self, x, t):
        """To be used in odeint"""
        # runs forward in time from 0, cannot just compute at arbitrary t
        """Lorenz 96 model."""
        # Compute state derivatives
        d = np.zeros(self.N)
        # First the 3 edge cases: i=1,2,N
        d[0] = (x[1] - x[self.N-2]) * x[self.N-1] - x[0]
        d[1] = (x[2] - x[self.N-1]) * x[0] - x[1]
        d[self.N-1] = (x[0] - x[self.N-3]) * x[self.N-2] - x[self.N-1]
        # Then the general case
        for i in range(2, self.N-1):
            d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
        # Add the forcing term
        d = d + self.F

        # Return the state derivatives
        return np.multiply(self.scaling_time_factor,d)

    def function(self,N,t):
        return self.interp_function(t)

    def prepare_f(self, times_array):
        """
        This needs to be run before the function 'function' can be used for this object.
        :param times_array: array of times at which to produce solution
        :return: states vector from run at times in times_array
        """
        t = times_array

        states = odeint(self.dfdt, self.state0, t)
        self.stored_state = states
        self.interp_function = scipy.interpolate.interp1d(times_array, self.stored_state.transpose(), kind='cubic')


        return states

class Colpitts_object():
    def __init__(self, noise=0, amplitude_magnifier = 1, scaling_time_factor=1.0):
        self.name = "I_Colpitts"
        self.alpha = 5.0
        self.gamma = 0.0797#0.08
        self.q = 0.6898#0.7
        self.eta = 6.273#6.3
        self.extra_descriptors = ('alpha=' + str(self.alpha) + ',gamma='+str(self.gamma) +',q='+str(self.q)+','+str(self.eta))
        self.noise = noise
        self.stored_state = np.array([])
        self.interp_function = None
        self.state0 = np.array([0.1, 0.1, 0.1])
        self.amplitude_magnifier = amplitude_magnifier
        self.scaling_time_factor = scaling_time_factor
        # Initial state (equilibrium)

    def dfdt(self, x, t):
        """To be used in odeint"""
        # runs forward in time from 0, cannot just compute at arbitrary t
        """Colpitts model."""
        # Compute state derivatives
        x1, x2, x3 = x  # Unpack the state vector
        dxdt = self.alpha * x2, -self.gamma * (x1 + x3) - self.q * x2, self.eta * (x2 + 1 - np.exp(-x1))  # Derivatives
        return np.multiply(self.scaling_time_factor,dxdt)

    def function(self,N,t):
        return self.amplitude_magnifier*self.interp_function(t)

    def prepare_f(self, times_array):
        """
        This needs to be run before the function 'function' can be used for this object.
        :param times_array: array of times at which to produce solution
        :return: states vector from run at times in times_array
        """
        t = times_array

        states = odeint(self.dfdt, self.state0, t)
        self.stored_state = states
        self.interp_function = scipy.interpolate.interp1d(times_array, self.stored_state.transpose(), kind='cubic')
        return states

class NaKL_object():
    def __init__(self, noise=0, amplitude_magnifier = 1, speedup_factor = 30, I_drive = 10):
        self.name = "I_NaKL"
        # NaKL Parameters
        self.gNa = 120
        self.ENa = 50
        self.gK = 20
        self.EK = -77
        self.gL = 0.3
        self.EL = -54.4
        self.Vm1 = -40.0
        self.dVm = 15
        self.taum0 = 0.1
        self.taum1 = 0.4
        self.Vh0 = -60
        self.dVh = -15
        self.tauh0 = 1
        self.tauh1 = 7
        self.Vn1 = -55
        self.dVn = 30
        self.taun0 = 1
        self.taun1 = 5
        self.extra_descriptors = ('')
        self.noise = noise
        self.stored_state = np.array([])
        self.interp_function = None
        self.state0 = np.array([-50, 0.4, 0.4, 0.4])
        self.amplitude_magnifier = amplitude_magnifier = 1
        self.speedup_factor = speedup_factor
        self.I_drive = I_drive # driving stimulus
        # Initial state (equilibrium)

    def dfdt(self, x, t):
        """To be used in odeint"""
        # runs forward in time from 0, cannot just compute at arbitrary t
        """NaKL"""
        dsdt = np.zeros_like(x)
        V, m, h, n = x

        # Equations of motion
        dsdt[0,] = self.gK * n ** 4 * (self.EK - V) + self.gL * (self.EL - V) + \
                   self.gNa * h * m ** 3 * (self.ENa - V) + self.I_drive
        dsdt[1,] = (-m + 0.5 * np.tanh((-self.Vm1 + V) / self.dVm) + 0.5) / \
                   (self.taum0 + self.taum1 * (1 - np.tanh((-self.Vm1 + V) / self.dVm) ** 2))
        dsdt[2,] = (-h + 0.5 * np.tanh((-self.Vh0 + V) / self.dVh) + 0.5) / \
                   (self.tauh0 + self.tauh1 * (1 - np.tanh((-self.Vh0 + V) / self.dVh) ** 2))
        dsdt[3,] = (-n + 0.5 * np.tanh((-self.Vn1 + V) / self.dVn) + 0.5) / \
                   (self.taun0 + self.taun1 * (1 - np.tanh((-self.Vn1 + V) / self.dVn) ** 2))
        return self.speedup_factor*dsdt

    def function(self,N,t):
        return self.amplitude_magnifier*self.interp_function(t)

    def prepare_f(self, times_array):
        """
        This needs to be run before the function 'function' can be used for this object.
        :param times_array: array of times at which to produce solution
        :return: states vector from run at times in times_array
        """
        t = times_array

        states = odeint(self.dfdt, self.state0, t)
        self.stored_state = states
        self.interp_function = scipy.interpolate.interp1d(times_array, self.stored_state.transpose(), kind='cubic')
        return states


class wavefile_object():
    def __init__(self, filename_load, filename_save, noise=0, times_array = None,
                 num_timesteps_in_window = 3, magnitude_multiplier = 1, time_scaling = 1,
                 input_dimension = 3):
        self.name = "wave_music"
        self.filename_load = filename_load # string; full path to filename + filename and extension
        self.filename_save = filename_save # string; full path to filename + filename and extension
        self.extra_descriptors = ('')
        self.noise = noise
        self.times_array = times_array
        self.t_final = None
        self.t_initial = None
        self.rate = None
        self.num_frames_for_wanted_seconds = None
        self.framespan = None
        self.num_frames_in_window = input_dimension
        self.data_channel = None
        self.interp_function = None
        self.fft_spectrum_t = None
        self.recovered_data = None
        self.start_frame = None
        self.data = None # Data which is directly imported and which is used as a basis for further reduction
        self.end_frame = None # Warning, this is a larger integer than you'd expect, because it needs to be large enough
        # for interpolation of the function to work over the whole integration interval (t_initial, t_final)

        self.rate_seg_div = 10
        self.magnitude_multiplier = magnitude_multiplier
        self.time_scaling = time_scaling
        self.input_dimension = input_dimension

    def load_wavefile(self):
        self.data, self.rate = sf.read(self.filename_load)
        num_frames = self.data.shape[0]
        # seconds_length_song = num_frames / self.rate
        # rate is frames per second
        # number of seconds to take from song
        self.t_initial = self.times_array[0]
        self.t_final = self.times_array[-1]
        self.num_frames_for_wanted_seconds = (self.t_final - self.t_initial) * self.rate
        self.data_channel = self.data[int(self.t_initial*self.rate):int(self.t_final*self.rate), 0]
        plt.plot(self.data_channel)
        plt.title("imported data")
        plt.show()


    def write_wavefile(self):
        sf.write(file=self.filename_save, data=self.recovered_data, samplerate=int(self.rate))

    def forward_FFT(self):
        print("Doing forward FFT")
        self.fft_spectrum_t = np.zeros((int(self.end_frame - self.start_frame), self.num_frames_in_window))
        for ind in range(int(self.num_frames_for_wanted_seconds-self.num_frames_in_window)):  # timewindows
            # for each timestep, store amplitudes of each frequency occurring over next num_timesteps_in_window
            temp = fft(self.data_channel[ind: ind + self.num_frames_in_window])
            self.fft_spectrum_t[ind] = temp

    # invert the FFT
    def inverse_FFT(self, returned_data):
        """
        :param returned_data: has dims (shape of data channel) = (large number,)
        :return: N/A
        """
        print("Doing inverse FFT")
        self.recovered_data = np.zeros(int(self.num_frames_for_wanted_seconds))
        self.recovered_data_t_function = np.zeros((np.shape(returned_data)[1]))
        # recovered_data has dims (shape of data channel) = (large number,)
        returned_data /= np.max(np.fabs(returned_data)) # normalize returned data so no nans appear from arctanh
        t_ind_in_window = np.shape(returned_data)[0] # same as number of frequencies
        for t_ind, t in enumerate(self.times_array[:-1]):  # timewindows
            # temp_1 = (-1 +np.power(10,np.arctanh(returned_data[:,t_ind])))#:t_ind+int(self.num_frames_in_window/self.rate)]))
            temp_1 = np.arctanh(returned_data[:,t_ind])#:t_ind+int(self.num_frames_in_window/self.rate)]))
            temp_2 = ifft(temp_1.flatten())
            self.recovered_data_t_function[t_ind] = np.real(np.sum(temp_2))
            # self.recovered_data_t_function[t_ind] = np.max(np.real(temp_2))
        # self.recovered_data_t_function = np.max(np.real(ifft(-1 +np.power(10,np.arctanh(returned_data)), axis=0)))
        print("Finished inverse FFT")
        print("Removing infinities")
        self.recovered_data_t_function = np.where(np.isinf(self.recovered_data_t_function), 0, self.recovered_data_t_function)
        self.recovered_data_t_function = np.where(np.isnan(self.recovered_data_t_function), 0, self.recovered_data_t_function)

        # Remove all elements more than 3 std devs away from median
        self.recovered_data_t_function[self.recovered_data_t_function > np.median(self.recovered_data_t_function) + 4*np.std(self.recovered_data_t_function)] = 0
        # self.recovered_data_t_function[:2500] = 0
        print(self.recovered_data_t_function)
        #normalization
        self.recovered_data_t_function /= np.max(self.recovered_data_t_function)
        print("Plotting recovered data")
        plt.plot(self.recovered_data_t_function)
        plt.ylim((-1,1))
        plt.title("recovered_data")
        plt.show()
        recovered_data_interp_function = scipy.interpolate.interp1d(self.times_array,
                                                                    self.recovered_data_t_function,
                                                                    kind='cubic',
                                                                    bounds_error=False,
                                                                    fill_value='extrapolate',
                                                                    assume_sorted='True')
        print("Saving recovered data in frames so that it can be written to wavefile.")
        for fr_ind, fr in enumerate(range(int(self.num_frames_for_wanted_seconds))):
            self.recovered_data[fr_ind] = recovered_data_interp_function(fr/self.rate)

        # remove last second of audio just because it seems to be problematic for some reason I don't understand
        self.recovered_data[-1000:] = 0

        # Remove overly large values from final array
        self.recovered_data[self.recovered_data > np.median(self.recovered_data) + 3*np.std(self.recovered_data)] = 0

        # Renormalize final array so that remaining data is of reasonable amplitude
        self.recovered_data /= np.max(self.recovered_data)


    # invert the FFT
    def inverse_FFT_ntime(self, returned_data):
        """
        :param returned_data: has dims (shape of data channel) = (large number,)
        :return: N/A
        """
        print("Doing inverse FFT")
        self.recovered_data = np.zeros(int(self.num_frames_for_wanted_seconds))
        self.recovered_data_t_function = np.zeros((np.shape(returned_data)[1]), dtype=complex)
        # recovered_data has dims (shape of data channel) = (large number,)
        returned_data /= np.max(np.fabs(returned_data)) # normalize returned data so no nans appear from arctanh
        t_ind_in_window = np.shape(returned_data)[0] # same as number of frequencies
        print("t_ind_in_window: "+str(t_ind_in_window))
        for t_ind, t in enumerate((self.times_array[:-1])[:-t_ind_in_window]):  # timewindows
            # temp_1 = (-1 +np.power(10,np.arctanh(returned_data[:,t_ind])))#:t_ind+int(self.num_frames_in_window/self.rate)]))
            temp_1 = np.arctanh(returned_data[:,t_ind])#:t_ind+int(self.num_frames_in_window/self.rate)]))
            temp_2 = ifft(temp_1.flatten())
            self.recovered_data_t_function[t_ind: t_ind+t_ind_in_window] += (temp_2)
            # self.recovered_data_t_function[t_ind] = np.max(np.real(temp_2))
        # self.recovered_data_t_function = np.max(np.real(ifft(-1 +np.power(10,np.arctanh(returned_data)), axis=0)))
        print("Finished inverse FFT")
        print("Removing infinities")
        self.recovered_data_t_function = np.where(np.isinf(self.recovered_data_t_function), 0, self.recovered_data_t_function)
        self.recovered_data_t_function = np.where(np.isnan(self.recovered_data_t_function), 0, self.recovered_data_t_function)

        # Remove all elements more than 3 std devs away from median
        self.recovered_data_t_function[np.fabs(self.recovered_data_t_function) > np.median(self.recovered_data_t_function) + 3*np.std(self.recovered_data_t_function)] \
            = np.median(self.recovered_data_t_function) + 1 * np.std(self.recovered_data_t_function)
        self.recovered_data_t_function[:2500] = 0
        # self.recovered_data_t_function[-2500:] = 0
        print(self.recovered_data_t_function)
        #normalization
        self.recovered_data_t_function /= np.max(self.recovered_data_t_function)
        print("Plotting recovered data")
        plt.plot(self.recovered_data_t_function)
        plt.ylim((-1,1))
        plt.title("recovered_data")
        plt.show()
        recovered_data_interp_function = scipy.interpolate.interp1d(self.times_array,
                                                                         np.real(self.recovered_data_t_function),
                                                                         kind='cubic')
        print("Saving recovered data in frames so that it can be written to wavefile.")
        for fr_ind, fr in enumerate(range(int(self.num_frames_for_wanted_seconds))):
            self.recovered_data[fr_ind] = recovered_data_interp_function(fr/self.rate)
        # self.recovered_data[-200:] = 0
        # Remove overly large values from final array
        self.recovered_data[np.fabs(self.recovered_data) > np.median(self.recovered_data) + 3*np.std(self.recovered_data)] = \
            np.median(self.recovered_data) + 1*np.std(self.recovered_data)

        # Renormalize final array so that remaining data is of reasonable amplitude
        self.recovered_data /= np.max(self.recovered_data)

    def function(self,N,t):
        # print("eval for t="+str(t))
        return self.magnitude_multiplier * (self.interp_function(t/self.time_scaling))

    def prepare_f(self, times_array):
        """
        Prepares frequency space data for interpolation.
        This needs to be run before the function 'function' can be used for this object.
        :param times_array: array of times at which to produce solution
        :return: not applicable
        """
        print("Preparing function "+str(self.name))
        t = times_array
        self.load_wavefile()
        self.start_frame = int(times_array[0] * self.rate) # frame corresponding to first time in times_array
        self.dt = (self.times_array[1]-self.times_array[0])
        self.end_frame =  int(times_array[-1] * self.rate + 2*self.dt*self.rate) # extra dt bit to make interpolation work well near wanted endpoint
        print("start_frame:" +str(self.start_frame))
        print("end_frame:" +str(self.end_frame))
        self.framespan = np.array(range(self.start_frame, self.end_frame))
        # self.framespan = np.array([fr for fr in range(self.num_frames_for_wanted_seconds)])
        print(self.framespan/self.rate)

        f, self.t, self.Sxx = spectrogram(self.data[:self.end_frame, 0], fs=1, nperseg=int(self.rate/self.rate_seg_div),
                                noverlap=int(self.rate/self.rate_seg_div*9.5/10))

        print("Sxx shape: "+str(self.Sxx.shape))
        print("f shape: "+str(f.shape))
        print()
        # print(self.t.shape)

        # print("Making plot")
        # plt.pcolormesh(self.t / self.rate, f, np.log10(self.Sxx), shading='gouraud')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()

        index_spacing = self.Sxx.shape[0]//self.input_dimension
        print("Frequency index spacing is "+str(index_spacing))
        list_of_indices = [i*index_spacing for i in range(self.input_dimension)]
        # Ensure spatial length is correct
        assert np.zeros(self.Sxx.shape[0])[list_of_indices].shape[0] == self.input_dimension
        print("Making a and b matrices")
        a = np.concatenate((np.array([0]),self.t/self.rate, self.end_frame*np.array([1])))
        b = np.vstack((np.zeros(self.Sxx.shape[0]),
                       self.Sxx.transpose(),
                       np.zeros(self.Sxx.shape[0]) ))[:,list_of_indices].transpose()

        print("Making interpolation function")
        self.interp_function = scipy.interpolate.interp1d(a,
                                                          np.tanh(b),
                                                          kind='cubic',
                                                          bounds_error=False,
                                                          fill_value='extrapolate',
                                                          assume_sorted='True') #TODO: Use tanh(log10())
                                                        # np.tanh(np.log10(np.fabs(b)+1)),
                                                        # kind='cubic') #TODO: Use tanh(log10())