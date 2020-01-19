import sys
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import to_categorical

#We build a CNN for the classification of the audio which comes as Mel-frequency cepstral coefficients

#specify hyper parameters
num_classes=2
input_shape=(20,1)#we have 20 Mel-frequency cepstral coefficients

batch_size = 1000
epochs = 20

#specify the model architecture

model = Sequential()
model.add(Conv1D(32, kernel_size=7, input_shape=input_shape))
model.add(Conv1D(64, kernel_size=7,padding='same'))
#model.add(MaxPooling1D(pool_size=2,padding='same'))
model.add(Conv1D(128, kernel_size=5,padding='same'))
model.add(MaxPooling1D(pool_size=2,padding='same'))
model.add(Conv1D(256, kernel_size=3,padding='same'))
model.add(MaxPooling1D(pool_size=2,padding='same'))
model.add(Conv1D(512, kernel_size=3,padding='same'))
model.add(MaxPooling1D(pool_size=2,padding='same'))
model.add(Conv1D(1028, kernel_size=3,padding='same'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

#load the data
X = np.load('audio_Muppets-02-01-01.npy')#[0:10000]
y = np.load('labels_audio_Muppets-02-01-01.npy')#[0:10000]

#make a trainig test split or...
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

#...use the whole data to train
#X_train = X
#y_train = y

#reshape the data: introduce a spartial dimension
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

#one hot encoding for labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))#dont use if there is no test data

score = model.evaluate(X_test, y_test, verbose=0)

model.save('model_audio.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])
