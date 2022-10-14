import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import *
from util.sound_util import *
from pylab import *


def extract_mfcc_feat(input_dir):
    (rate, sig) = wav.read(input_dir)
    signal = end_point_detection(input_dir)
    for i in range(len(signal)):
        signal[i] = int(round(signal[i]))
    signal = np.array(signal)
    frame_len = 0.02
    frame_step = 0.01
    frame_size = int(round(frame_len * rate))
    mfcc_feat = mfcc(signal, samplerate=rate, winlen=frame_len, winstep=frame_step, nfft=frame_size)
    return mfcc_feat


def get_all_mfcc_feat():
    training_data = []
    test_data = []
    training_path = "../data/training data"
    test_path = "../data/test data"
    for file in os.listdir(training_path):
        if file.split('.')[1] == "wav":
            training_data.append(extract_mfcc_feat(os.path.join(training_path, file)))
    for file in os.listdir(test_path):
        if file.split('.')[1] == "wav":
            test_data.append(extract_mfcc_feat(os.path.join(test_path, file)))
    print("!")

if __name__ == '__main__':
    # extract_mfcc_feat("C:\\Users\\24111\\PycharmProjects\\voice-recognition\\data\\training data\\s1a.wav")
    get_all_mfcc_feat()
