import matplotlib.pyplot as plt
import numpy as np
import os.path
from neo import AxonIO
from neo import NeoMatlabIO

import quantities as pq


def save_fig_with_makedir(figure, save_location):
    """
    input: matplotlib figure fig, and string "save_location" (example: directory1/directory2/..../filename.ext)
    Creates all directories within save_location if they don't yet exist, then saves figure fig to save_location
    """
    if "/" in save_location:
        last_slash_index = save_location.rfind('/') #finds last location of "/" in save_location
    directory = save_location[:last_slash_index]
    filename  = save_location[last_slash_index:]
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if os.path.isdir(directory):
        figure.savefig(directory + str(filename), bbox_inches='tight')

def save_txt_with_makedir(data, save_location):
    """
    input: text data, and string "save_location" (example: directory1/directory2/..../filename.ext)
    Creates all directories within save_location if they don't yet exist, then saves data to save_location
    """
    if "/" in save_location:
        last_slash_index = save_location.rfind('/') #finds last location of "/" in save_location
    directory = save_location[:last_slash_index]
    filename  = save_location[last_slash_index:]
    if not os.path.isdir(directory):
        print("Creating "+str(directory))
        os.makedirs(directory)
    if os.path.isdir(directory):
        print("Saving to "+str(directory))
        print("Saving: "+str(directory + str(filename)))
        np.savetxt(directory + str(filename), data)

def save_dict_with_makedir(data, save_location):
    """
    input: dictionary data, and string "save_location" (example: directory1/directory2/..../filename.ext)
    Creates all directories within save_location if they don't yet exist, then saves data to save_location
    """
    if "/" in save_location:
        last_slash_index = save_location.rfind('/') #finds last location of "/" in save_location
    directory = save_location[:last_slash_index]
    filename  = save_location[last_slash_index:]
    if not os.path.isdir(directory):
        print("Creating "+str(directory))
        os.makedirs(directory)
    if os.path.isdir(directory):
        print("Saving to "+str(directory))
        print("Saving: "+str(directory + str(filename)))
        np.save(directory + str(filename), data)


def save_and_or_display_plot(figure, a_str, save_location):
    """
    If "a_str" == "save", then the figure will be saved at string save_location, and all directories in the
    save_location string will be created if they do not yet exist.
    """
    if "save" in a_str.lower():
        save_fig_with_makedir(figure,save_location)
    if "display" not in a_str.lower():
        plt.close(figure)

def save_text(data, a_str, save_location):
    """
    If "a_str" == "save", then the data will be saved at string save_location, and all directories in the save_location
    string will be created if they do not yet exist.
    """
    if "save" in a_str.lower():
        save_txt_with_makedir(data, save_location)

def glob_extension_case_string_builder(input_extension):
    """
    Returns a string to later be used in glob's recursive file iteration.
    Basically takes a string and allows "case insensitive" searches for it
    """
    return_str = ""
    if "." in input_extension and input_extension[0]==".":
        input_extension = input_extension[1:]
    for i, character in enumerate(input_extension):
        return_str += ("["+str(character.lower())+str(character.upper())+"]")
    return return_str

def give_name_if_included_in_path(path_str, name_list):
    """
    Returns whichever element of name_list is present in the path string
    """
    for a_name in name_list:
        if a_name in path_str:
            return a_name
    return ""

def load_and_prepare_abf_or_mat_data(neuron_directory, a_filename, directory_to_store_txt_data, file_extension):
    """
    Converts data from an atf file to three column (VIt) format, and then saves the .txt file
    """
    junction_potential = pq.Quantity(11.6,
                                     'mV')  # measured at 32 C (NOTE: I'm not completely sure if this applies to all measurements of CM in these directories. I should ask Prof. Meliza)
    # open the file
    # print("Preparing to use file " + a_filename)
    if file_extension.lower() in "mat":
        fp = NeoMatlabIO(filename=neuron_directory + a_filename)
    if file_extension.lower() in "abf":
        fp = AxonIO(filename=neuron_directory + a_filename)
    # read the data. There is only one block in each file
    block = fp.read_block()
    # Neo calls sweeps "segments"
    sweep = block.segments[0]
    # each sweep (here, block.segments[0]) has one or more channels. Channel 0 is always V and channel 1 is always I.
    V = (sweep).analogsignals[0] - junction_potential
    I = (sweep).analogsignals[1]
    t = (V.times - V.t_start)  # measured in ms
    Current_unit = (str(I.units)[-2:]).strip() # usually pA
    Voltage_unit = (str(V.units)[-2:]).strip() # usually mV
    Time_unit = (str(t.units)[-2:]).strip() # usually seconds

    TT = ((t[1] - t[0])).magnitude  # this is milliseconds
    V_and_I_arr = np.concatenate((V.magnitude, I.magnitude), axis=1)
    t_arr = np.array([t]).transpose()
    # make directory for storing data in .txt form if it doesn't exist yet

    save_text(data=np.concatenate((V_and_I_arr, t_arr), axis=1), a_str="save",
              save_location=directory_to_store_txt_data + str(a_filename[:-4]) + "_VIt.txt")

    return [Current_unit, Voltage_unit, Time_unit]