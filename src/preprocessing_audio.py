import sys
import librosa
import numpy as np
import math
import matplotlib.pyplot as plt
import librosa.display

#this script is used to convert the wav files to numbers and to introduce the labels needed for the classification task

#do the following steps for each .wav file
IMAGE_PATH = 'Muppets-02-01-01'
file_path = "Audio\\" + IMAGE_PATH+ ".wav"

#get the Mel-frequency cepstral coefficients
wave, sr = librosa.load(file_path, mono=True, sr=None)
duration = librosa.get_duration(wave, sr=sr)
mfcc = librosa.feature.mfcc(wave, sr=sr)

#introduce labels:
kermit_present = np.loadtxt(IMAGE_PATH+' Kermit/filenames.txt')
kermit_present = kermit_present.astype(int)
labels = np.zeros(np.shape(mfcc)[1])

#the labeling was made for two frames per second, the new data has much more data points per second
#addapt the labeling from 2FPS to the discretization of the mfcc
step = np.shape(mfcc)[1]/math.ceil(duration*2)

for i in kermit_present:
    labels[math.floor((i-1)*step):math.ceil((i)*step)] = 1

#pad all data sets to the same length
pad_width = 133333 #fix dimension

mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width - mfcc.shape[1])), mode='constant')
labels = np.pad(labels, pad_width=((0, pad_width - labels.shape[0])), mode='constant')

#cut the decimals to save memory and transpose mfcc
audio = np.around(mfcc.T,decimals=2)

#plot the mfcc
plt.figure(figsize=(6, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

#save the mfcc and the labeling
np.save('audio_'+IMAGE_PATH+'.npy', audio)
np.save('labels_audio_'+IMAGE_PATH+'.npy', labels)




