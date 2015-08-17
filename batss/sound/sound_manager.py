import numpy as np
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
from .sound import Sound


# np.set_printoptions(threshold=np.nan)

class SoundManager:
    def __init__(self):
        pass

    def read_file(self, file_name):
        dotwav = os.getcwd() + "/" + file_name
        # rate is the sampling rate, sample is the array of samples
        (rate, sample) = wav.read(dotwav)
        if sample.ndim == 1:
            return Sound(sample, rate).normalize()
        else:
            s1 = Sound(sample[:, 0], rate).normalize()
            s2 = Sound(sample[:, 1], rate).normalize()
            return s1, s2

    def plot(self, sound):
        if type(sound) == tuple:
            for i in range(0, 2):
                time_array = np.arange(0, sound[i].sample.size, 1)
                time_array = time_array / sound[i].rate
                time_array = time_array * 1000  # scale to milliseconds
                plt.subplot(2, 1, i+1)
                plt.plot(time_array, sound[i].sample, color='red')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (ms)')
            plt.show()
        else:
            time_array = np.arange(0, sound.sample.size, 1)
            time_array = time_array / sound.rate
            time_array = time_array * 1000  # scale to milliseconds
            plt.plot(time_array, sound.sample, color='red')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (ms)')
            plt.show()

if __name__ == "__main__":
    s = SoundManager()
    sound = s.read_file("WAV/X_linear.wav")
    # sound = s.read_file("CallHangup")
    s.plot(sound)
