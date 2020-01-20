import cv2
import glob
import numpy as np
import os.path as path
from scipy import misc
import sys

#this script is used to convert the video files to numbers and to introduce the labels needed for the classification task

#do the following steps for each video file
#convert video to images and save them (by Khush Patel):
clip = 'Muppets-03-04-03.avi'
vidcap = cv2.VideoCapture(clip)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(clip+"/image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 0.5 #we consider two frames per second
count=1
success = getFrame(sec)
#while success:
 #   count = count + 1
  #  sec = sec + frameRate
   # sec = round(sec, 2)
    #success = getFrame(sec)

#manually sort the images where Kermit is present and obtain a text file (filenames.txt) with their numbers

#Load the images as numpy arrays:
IMAGE_PATH = 'Muppets-02-04-04'
file_paths = glob.glob(path.join(IMAGE_PATH, '*.jpg'))
images = [misc.imread(path) for path in file_paths]
images = np.asarray(images)
# Get image size
print(np.shape(images))
#pad the images to 544 x 720
if np.shape(images) != (np.shape(images)[0], 544, 720, 3):
    images = np.pad(images, [(0, 0), (0, 544 - images.shape[1]), (0, 720 - images.shape[2]), (0, 0)], mode='constant', constant_values=0)

np.save('images.npy', images)

#introduce labels:
kermit_present = np.loadtxt(IMAGE_PATH+' Kermit/filenames.txt')
kermit_present = kermit_present.astype(int)
labels = np.zeros(np.shape(images)[0])
labels[kermit_present-1] = 1
np.save('labels.npy', labels)
print(labels)


