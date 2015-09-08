import numpy as np
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
from .sound import Sound
import math

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

    def plot(self, sound): # sound should be a list of sound objects
        count = 0
        for j in range(0, len(sound)):
            if type(sound[j]) == tuple:
                for i in range(0, len(sound[j])):
                    time_array = np.arange(0, sound[j][i].sample.size, 1)
                    time_array = time_array / sound[j][i].rate
                    time_array = time_array * 1000  # scale to milliseconds
                    count += 1
                    plt.subplot(len(sound), 2, count)
                    plt.plot(time_array, sound[j][i].sample, color='gray')
            else:
                time_array = np.arange(0, sound[j].sample.size, 1)
                time_array = time_array / sound[j].rate
                time_array = time_array * 1000  # scale to milliseconds
                count += 1
                if (count%2==0):
                    count += 1
                plt.subplot(len(sound), 2, count)
                count += 1
                plt.plot(time_array, sound[j].sample, color='gray')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (ms)')
        plt.show()

    def fft(self, sample):
        frequencies = np.fft.fft(sample)
        amplitudes = np.abs(frequencies)
        angles = np.angle(frequencies)
        print(sample)
        return amplitudes, angles

    def ifft(self, amplitudes, angles):
        sample = np.real(np.fft.ifft(amplitudes * np.exp(1j * angles)))
        print(sample)
        return sample

    def plot_fft(self, sound): # sound should be a list of sound objects
        count = 0
        for j in range(0, len(sound)):
                count += 1
                trans_sound = np.fft.fft(sound[j].sample)
                n = len(sound[j].sample)
                nUniquePts = math.ceil((n+1)/2.0)
                trans_sound = trans_sound[0:nUniquePts]
                trans_sound = abs(trans_sound)
                trans_sound = trans_sound / float(n)
                if n % 2 > 0:
                    trans_sound[1:len(trans_sound)] = trans_sound[1:len(trans_sound)] * 2
                else:
                    trans_sound[1:len(trans_sound) - 1] = trans_sound[1:len(trans_sound)-1] * 2
                freqArray = np.arange(0, nUniquePts, 1.0) * (sound[j].rate / n)
                plt.subplot(len(sound), 1, count)
                plt.plot(freqArray/1000, 10*np.log10(trans_sound), color='gray')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Power (dB)')
        plt.show()

    def save_file(self, sound, file_name):
        dotwav = os.getcwd() + "/" + file_name
        wav.write(dotwav, sound.rate, sound.sample)

if __name__ == "__main__":
    s = SoundManager()
    sound = []
    #sound.append(s.read_file("sound/WAV/X_linear.wav"))
    sound.append(s.read_file("sound/WAV/CallHangup.wav"))
    #sound.append(s.read_file("sound/WAV/X_linear.wav"))
    #sound = s.read_file("WAV/CallHangup.wav")
    a = s.fft(sound[0].sample)
    sam = s.ifft(a[0],a[1])
    play = Sound(sam, sound[0].rate)
    sound.append(play)
    s.plot_fft(sound)
    #s.plot_fft(play)
    #s.plot_fft(sound)
    #s.save_file(sound[1], "sound/WAV/generated1.wav")
