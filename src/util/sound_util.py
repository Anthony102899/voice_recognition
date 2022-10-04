import sys
import os
import wave
import struct
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
    print(amplitude)
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
    time = np.arange(0, nframes)/framerate
    plt.plot(time, amplitude)
    plt.show()


# return the T1, T2 that denote for the start and the end of speech voice respectively.
def end_point_detection(input_dir):
    amplitude, nchannels, nframes, sample_width, framerate = voice_input(input_dir)
    time = np.arange(0, nframes)/framerate
    energy = zeros(nframes)
    for i in range(nframes):
        energy[i] = amplitude[i] * amplitude[i]

    frame_length = 0.02
    non_overlapping = 0.01

    frame_size = round(framerate * frame_length)
    non_overlapping_size = round(framerate * non_overlapping)
    overlapping_size = round(framerate * (frame_length - non_overlapping))
    frames = []
    counter = 0
    while counter < nframes:
        frames.append(amplitude[counter: counter + frame_size])
        counter += non_overlapping_size
    frame_energy = zeros(len(frames))
    counter = 0
    for i in range(len(frames)):
        frame_energy[i] = sum(energy[counter: counter + frame_size]) * 0.5
        counter += non_overlapping_size
    print(frame_energy)


if __name__ == "__main__":
    # voice_plot(os.path.join(TRAINING_DIR, "s1a.wav"))
    end_point_detection(os.path.join(TRAINING_DIR, "s1a.wav"))
