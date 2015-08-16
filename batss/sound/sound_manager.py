import numpy as np
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
from sound import Sound


# np.set_printoptions(threshold=np.nan)

class SoundManager:
    def __init__(self):
        pass

    def read_file(self, fileName):
        folder = os.getcwd() + "/WAV"
        dotwav = folder + "/" + fileName + ".wav"
        (rate, sample) = wav.read(dotwav)  # rate is the sampling rate, sample is the array of samples
        if sample.ndim == 1:
            return Sound(sample, rate).normalize()
        else:
            return Sound(sample[0], rate).normalize(), Sound(sample[1], rate).normalize()

    def plot(self, sound):
        if type(sound) == tuple:
            for i in range(0, 2):
                timeArray = np.arange(0, sound[i].sample.size, 1)
                timeArray = timeArray / sound[i].rate
                timeArray = timeArray * 1000  # scale to milliseconds
                plt.subplot(2,1,i+1)
                plt.plot(timeArray, sound[i].sample, color='red')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (ms)')
            plt.show()
        else:
            timeArray = np.arange(0, sound.sample.size, 1)
            timeArray = timeArray / sound.rate
            timeArray = timeArray * 1000  # scale to milliseconds
            plt.plot(timeArray, sound.sample, color='red')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (ms)')
            plt.show()

s = SoundManager()
sound = s.read_file("X_linear")
#sound = s.read_file("CallHangup")
s.plot(sound)
