import sys
import librosa
import numpy as np
import math
import matplotlib.pyplot as plt
import librosa.display
import glob
import os.path as osp
import  librosa as lr
# Define the paths
base = "../data/audio/"
processed_dir = "../../processed_img/"

muppet_dirs = ["Muppets-02-01-01" , "Muppets-03-04-03"]
test_dir = ["Muppets-02-04-04"]

pattern = "*.mp4"
num_mffcs = 40
input_shape = (num_mffcs,44)


def save_chunks(dirs:[str], mode:str):
    chunks = []
    for dir in dirs:
        # do the following steps for each .wav file
        # ../..data/audio/Muppets-02-01-01/Muppets-02-01-01.wav
        kermit_chunks = base + dir + "/kermit/" + pattern

        count = 0
        print("process ", kermit_chunks)
        for file_chunk in glob.iglob(kermit_chunks):
            # print(file_chunk)
            # load audio
            # data = audio time series
            # sr = sampling rate of y
            data, sr = lr.load(file_chunk)
            S = np.abs(lr.stft(data))
            # pitches, magnitudes = librosa.piptrack(S=S, sr=sr)
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=num_mffcs)

            # print(count, " ", mfccs.shape)

            if mfccs.shape[1] < input_shape[1]:  # need to add 0 column
                padd = np.zeros(input_shape)
                padd[:, :mfccs.shape[1]] = mfccs
                chunks.append(padd)
            else:
                chunks.append(mfccs)

            count += 1
            if count % 100 == 0:
                print("count ", count)

        nkermit_chunks = base + dir + "/not_kermit/" + pattern
        print("process not_kermits")
        count = 0
        for file_chunk in glob.iglob(nkermit_chunks):
            data, sr = lr.load(file_chunk)
            S = np.abs(lr.stft(data))
            # pitches, magnitudes = librosa.piptrack(S=S, sr=sr)
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=num_mffcs)

            if mfccs.shape[1] < input_shape[1]:  # need to add 0 column
                padd = np.zeros(input_shape)
                padd[:, :mfccs.shape[1]] = mfccs
                chunks.append(padd)
            else:
                chunks.append(mfccs)

            count += 1
            if count % 100 == 0:
                print("count ", count)

    chunks_arr = np.asarray(chunks)
    print(chunks_arr.shape)
    np.save(mode+"chunks", chunks_arr)


if __name__ == '__main__':
    # this script is used to convert the wav files to numbers and to introduce the labels needed for the classification task
    save_chunks(muppet_dirs, "train")
    save_chunks(test_dir,"test")