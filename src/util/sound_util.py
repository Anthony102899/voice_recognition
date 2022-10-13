import sys
import os
import wave
import struct

import matplotlib.pyplot as plt
from scipy import *
import matplotlib
from pylab import *
import numpy as np

matplotlib.use('TkAgg')

TRAINING_DIR = "C:\\Users\\24111\\PycharmProjects\\voice-recognition\\data\\training data"
TESTING_DIR = "C:\\Users\\24111\\PycharmProjects\\voice-recognition\\data\\test data"


def voice_input(input_dir):
    input_file = wave.open(input_dir, "rb")
    nchannels = input_file.getnchannels()
    nframes = input_file.getnframes()
    sample_width = input_file.getsampwidth()
    framerate = input_file.getframerate()

    amplitude = zeros(nframes)
    for i in range(nframes):
        value = input_file.readframes(1)
        amplitude[i] = struct.unpack('h', value)[0]
    input_file.close()
    return amplitude, nchannels, nframes, sample_width, framerate


'''
params:
    input_dir: String. The absolute path of .wav file.

returns:
    An image of the voice in time domain.
'''


def voice_plot(input_dir):
    amplitude, nchannels, nframes, sample_width, framerate = voice_input(input_dir)
    time = np.arange(0, nframes) / framerate
    plt.plot(time, amplitude)
    plt.show()


# return the T1, T2 that denote for the start and the end of speech voice respectively.
def end_point_detection(input_dir):
    amplitude, nchannels, nframes, sample_width, framerate = voice_input(input_dir)
    energy = zeros(nframes)
    for i in range(nframes):
        energy[i] = amplitude[i] * amplitude[i]

    frame_length = 0.02
    non_overlapping = 0.01

    # initialize the frames according to the frame length and overlapping size.
    frame_size = round(framerate * frame_length)
    non_overlapping_size = round(framerate * non_overlapping)
    overlapping_size = round(framerate * (frame_length - non_overlapping))
    frames = []
    counter = 0
    while counter < nframes:
        frames.append(amplitude[counter: counter + frame_size])
        counter += non_overlapping_size

    # calculate the energy level for each frame.
    frame_energy = zeros(len(frames))
    counter = 0
    for i in range(len(frames)):
        frame_energy[i] = sum(energy[counter: counter + frame_size]) * 0.5
        counter += non_overlapping_size
    energy_bound = max(frame_energy) / 20  # the energy lower bound is the 1/20 of the highest energy.
    # time = np.arange(0, len(frames))
    # plt.plot(time, frame_energy)
    # plt.show()
    # calculate the zero crossing rate for each frame.
    zero_crossing = zeros(len(frames))
    for i in range(len(frames)):
        for j in range(1, len(frames[i])):
            if (frames[i][j] < 0 and frames[i][j - 1] > 0) or (frames[i][j] > 0 and frames[i][j - 1] < 0):
                zero_crossing[i] += 1
    zc_bound = 20  # empirical try
    # time = np.arange(0, len(frames))
    # plt.plot(time, zero_crossing)
    # plt.show()
    # search for the starting point
    starting_point = len(frames) - 1
    for i in range(0, len(frames) - 2):
        # if frame_energy[i] >= energy_bound and frame_energy[i + 1] >= energy_bound and frame_energy[
        #     i + 2] >= energy_bound and zero_crossing[i] > zc_bound and zero_crossing[i + 1] > zc_bound and \
        #         zero_crossing[i + 2] > zc_bound:
        if frame_energy[i] >= energy_bound and frame_energy[i + 1] >= energy_bound and frame_energy[i + 2] >= energy_bound:
            starting_point = i
            break
    print(starting_point)
    starting_time = 0.01 * starting_point
    ending_point = starting_point
    for i in range(starting_point, len(frames)):
        if starting_point < 2:
            continue
        else:
            if frame_energy[i] < energy_bound and frame_energy[i - 1] < energy_bound and frame_energy[i - 2] < energy_bound:
                ending_point = i
                break
    print(ending_point)
    ending_time = 0.01 * ending_point
    # plot the start and ending point in the figure
    time = np.arange(0, nframes) / framerate
    plt.plot(time, amplitude)
    plt.hlines(amplitude, starting_time, starting_time + 0.01, colors="r")
    plt.hlines(amplitude, ending_time, ending_time + 0.01, colors="r")

    plt.hlines(amplitude, starting_time + 0.04, starting_time + 0.045, colors="g")
    plt.hlines(amplitude, starting_time + 0.06, starting_time + 0.065, colors="g")
    plt.show()




if __name__ == "__main__":
    # voice_plot(os.path.join(TRAINING_DIR, "s1a.wav"))
    end_point_detection(os.path.join(TRAINING_DIR, "s1a.wav"))
