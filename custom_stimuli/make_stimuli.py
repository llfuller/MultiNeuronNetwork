import numpy as np
import matplotlib.pyplot as plt
import externally_provided_currents


def Fourier_Power_Spectrum_plot_and_save(data, name, sampling_rate, mean, xlim=175):
    # Training Current with no modifications
    fourier_transform = np.fft.rfft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))
    delta_freq = frequency[3] - frequency[2]

    plt.figure()
    freq_without_0_index = frequency[1:]
    normalized_power_spec_without_0_index = power_spectrum[1:] / np.max(np.abs(power_spectrum[1:]))

    plt.plot(freq_without_0_index,
             normalized_power_spec_without_0_index / np.max(np.abs(normalized_power_spec_without_0_index)))
    plt.title("Power(freq) of training current from ("+str(name)+")")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Power (1.0= max from spectrum times [1:])")
    plt.xlim(0, 175)
    plt.savefig("range="+str(mean)+"pA/Power spectrum of training current from ("+str(name)+").png")
    # plt.show()
    np.savetxt("range="+str(mean)+"pA/Power spectrum of training current from ("+str(name)+") range="+str(mean)+".atf",
               np.column_stack((freq_without_0_index, normalized_power_spec_without_0_index)))
    plt.close()

def Data_plot_and_save(data, t_arr, mean, name):
    plt.figure()
    plt.plot(t_arr, data)
    plt.title('Current vs Time')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Current (pA)")
    plt.savefig("range="+str(mean)+"pA/Current vs Time ("+str(name)+").png")
    # np.savetxt("range="+str(mean)+"pA/"+name+"range="+str(mean)+"_(I).atf",
    #            np.concatenate((t_arr[:, np.newaxis], data[:, np.newaxis]), axis=1))
    np.savetxt("range="+str(mean)+"pA/"+name+"range="+str(mean)+"_(I).atf", data)
    plt.close()

def set_mean(arr, new_mean=1):
    # return np.divide(arr,np.mean(arr))*new_mean
    # return np.divide(arr,np.max(arr))*new_mean
    return None
def set_range(arr, lower_bound, upper_bound):
    return np.divide(arr+np.abs(np.min(arr)),np.max(arr)+np.abs(np.min(arr)))*\
           (upper_bound + np.abs(lower_bound)) - np.abs(lower_bound)

# Choose whether to convert to other units for output
# convert_pA_to_nA = False
# convert_s_to_ms = False
# conversion_pA_to_nA = np.power(0.001,int(convert_s_to_ms))
# conversion_s_to_ms = np.power(1000.0,int(convert_s_to_ms))

# for mean_value in [200,300]:
for dilation_factor in [0.1,0.2,0.5,1.0,4.0,10.0]:#[0.2,0.5,1.0,2.0,4.0,8.0]:#0.2, 1.0, 2.0, 4.0, 6.0]: # dilation_factor of 1 = good and normal Fourier Power Spectrum.
    print("Dilation factor:"+str(dilation_factor))
    for max_value in [100,200,300,400,500,600,700]: # max value of current range in pAmps

        # --------------- Current used on Meliza Data --------------
        mean_value=max_value
        # Current vs Time
        print("Initial load for max_value="+str(max_value))
        VIT = np.loadtxt("2014_12_11_0017_VIt.txt")
        t_arr = VIT[:,2]
        I_arr = VIT[:,1]

        I_2014_12_11_0017 = set_range(I_arr, -100, max_value)

        Data_plot_and_save(I_2014_12_11_0017, t_arr, mean=mean_value, name= "I_920061fe_2014_12_11_0017")

        timestep = 0.00002 #seconds
        sampling_rate = 1.0/(timestep)
        plt.figure()
        plt.plot(t_arr,I_2014_12_11_0017)
        plt.title('Reloaded Current vs Time')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Current (pA)")
        Fourier_Power_Spectrum_plot_and_save(data=I_2014_12_11_0017, name="I_920061fe_2014_12_11_0017",
                                             mean=mean_value, sampling_rate=sampling_rate)

        #====================================================================================================
        # Other currents
        #==================================================================================================
        timestep = 0.00002 #seconds
        sampling_rate = 1.0/(timestep)
        total_time = 15 #seconds
        t_arr = np.arange(start=0,stop=total_time,step=timestep)

        # --------------- L63 --------------
        plot_3D = False
        scaling_time = dilation_factor*22.0#200
        L63_obj = externally_provided_currents.L63_object(scaling_time_factor=scaling_time)
        I_L63_x, I_L63_y, I_L63_z = L63_obj.prepare_f(t_arr).T
        I_L63_x = set_range(I_L63_x, -100, max_value)
        I_L63_y = set_range(I_L63_y, -100, max_value)
        I_L63_z = set_range(I_L63_z, -100, max_value)

        Data_plot_and_save(I_L63_x, t_arr, mean=mean_value, name="I_L63_x_time_dilation="+str(dilation_factor))

        if plot_3D == True:
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(I_L63_x[0:-1:10], I_L63_y[0:-1:10], I_L63_z[0:-1:10], c=t_arr[0:-1:10], cmap='winter')
            # plt.show()

        Data_plot_and_save(I_L63_x, t_arr, mean=mean_value, name= "I_L63_x_time_dilation="+str(dilation_factor))
        # Data_plot_and_save(I_L63_y, t_arr, mean=mean_value, name= "I_L63_y_time_dilation="+str(dilation_factor))
        # Data_plot_and_save(I_L63_z, t_arr, mean=mean_value, name= "I_L63_z_time_dilation="+str(dilation_factor))

        Fourier_Power_Spectrum_plot_and_save(data=I_L63_x, name="I_L63_x_time_dilation="+str(dilation_factor),
                                             mean=mean_value, sampling_rate=sampling_rate)

        # --------------- Colpitts --------------
        plot_3D = False
        scaling_time = dilation_factor*150.0#200
        colp_obj = externally_provided_currents.Colpitts_object(scaling_time_factor=scaling_time)

        print(colp_obj.prepare_f(t_arr).shape)
        I_colpitts_x, I_colpitts_y, I_colpitts_z = colp_obj.prepare_f(t_arr).T

        I_colpitts_x = set_range(I_colpitts_x, -100, max_value)
        I_colpitts_y = set_range(I_colpitts_y, -100, max_value)
        I_colpitts_z = set_range(I_colpitts_z, -100, max_value)

        Data_plot_and_save(I_colpitts_x, t_arr, mean=mean_value, name= "I_colpitts_x_time_dilation="+str(dilation_factor))
        Fourier_Power_Spectrum_plot_and_save(data=I_colpitts_x, name="I_colpitts_x_time_dilation="+str(dilation_factor),
                                             mean=mean_value, sampling_rate=sampling_rate)
        Data_plot_and_save(I_colpitts_y, t_arr, mean=mean_value, name= "I_colpitts_y_time_dilation="+str(dilation_factor))
        Fourier_Power_Spectrum_plot_and_save(data=I_colpitts_y, name="I_colpitts_y_time_dilation="+str(dilation_factor),
                                             mean=mean_value, sampling_rate=sampling_rate)
        Data_plot_and_save(I_colpitts_z, t_arr, mean=mean_value, name= "I_colpitts_z_time_dilation="+str(dilation_factor))
        Fourier_Power_Spectrum_plot_and_save(data=I_colpitts_z, name="I_colpitts_z_time_dilation="+str(dilation_factor),
                                             mean=mean_value, sampling_rate=sampling_rate)


        # --------------- L96 --------------
        scaling_time = dilation_factor*22.0
        L96_obj = externally_provided_currents.L96_object(scaling_time_factor=scaling_time)

        print(L96_obj.prepare_f(t_arr).shape)
        L96_x1 = (L96_obj.prepare_f(t_arr).T)[0]
        L96_x1[:4000] = 0 # remove transients
        L96_x1 = set_range(L96_x1, -100, max_value)

        Data_plot_and_save(L96_x1, t_arr, mean=mean_value, name= "I_L96_x1_time_dilation="+str(dilation_factor))
        Fourier_Power_Spectrum_plot_and_save(data=L96_x1, name="I_L96_x1_time_dilation="+str(dilation_factor),
                                             mean=mean_value, sampling_rate=sampling_rate)
