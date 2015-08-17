import numpy as np
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt

#matplotlib.use("TKAgg")

#np.set_printoptions(threshold=np.nan)

class soundMgr:
	def __init__(self):
		pass

	def readFile(self,fileName):
		folder = os.getcwd() + "/WAV"
		dotwav = folder + "/" + fileName + ".wav"
		(self.rate, self.sample) = wav.read(dotwav)	#rate is the sampling rate, sample is the array of samples

	def norm(self): #Normalization
		self.sample = self.sample / (2.**15) #Because sample is int16 type i.e, value ranges from -2^15 to 2^15

	def dispAttributes(self):
		print(self.rate)
		print(self.sample)
		#print(self.sample.shape)
	
	def plot(self):
		timeArray = np.arange(0, self.sample.size, 1)
		timeArray = timeArray / self.rate
		timeArray = timeArray * 1000  #scale to milliseconds
		plt.plot(timeArray, self.sample, color='k')
		plt.ylabel('Amplitude')
		plt.xlabel('Time (ms)')
		plt.show()
		

s = soundMgr()
s.readFile("CallHangup")
s.norm()
s.dispAttributes()
s.plot()
