import sys
import numpy as np
from keras.models import load_model
import math

#the function to test if Kermit is present in a video sequence:
#start and end in seconds for the boundaries of the sequence, video is the name of the video 
#model_images.h5 and model_audio.h5 have to be in the same path as this script

def ishethere(start,end,video):
    if end <= start:
        return('The end point has to be before the start point.')

    #Images
    model_images = load_model('model_images.h5')

    X = np.load('images_'+video+'.npy')[start:end] #multiply start and end by the FPS, e.g 2FPS -> [start*2,end*2]
    #make predictions
    y_pred = model_images.predict(X)
    max_value_images = np.max(y_pred.T, axis=1)[1]
    print(max_value_images)
    # data type for output
    max_value_images_str=str(max_value_images.astype(np.float))

    #Audio
    X = np.load('audio_'+video+'.npy')[math.floor(start*86.12):math.ceil(end*86.12)]
    X = np.expand_dims(X, axis=2)

    model_audio = load_model('model_audio.h5')
    # make predictions
    y_pred = model_audio.predict(X)
    max_value_audio = np.max(y_pred.T, axis=1)[1]
    #data type for output
    max_value_audio_str = str(max_value_audio.astype(np.float))

    #failed accuracy
    fail_acc = ((1 - max_value_images) + (1 - max_value_audio)) / 2
    fail_acc_str = str(fail_acc.astype(np.float))

    #decision policy: basically if one of the classifiers is sure with confident >0.5 on any discrete data point, we output yes
    if max_value_audio>=0.5 and max_value_images>=0.5:
        return ('Kermit is present with ' + max_value_audio_str + 'and'+ max_value_images_str+'accuracy by hearing and seeing him respectively')
    elif max_value_audio>=0.5:
        return('Kermit is present with '+ max_value_audio_str +' accuracy by hearing him')
    elif max_value_images>=0.5:
        return('Kermit is present with '+ max_value_images_str+' accuracy by seeing him')
    else:
        return('Kermit is not present with an accuracy of '+fail_acc_str)


print(ishethere(0,5,'Muppets-02-01-01'))
