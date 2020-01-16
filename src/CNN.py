import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

#fixed input shape (height, width and RGB of a frame) 
input_shape=(544,720,3)
#two classes present
num_classes=2

#for training:
#epochs is the number how often the training data is feed into the network
#split each epoch in smaller batches 
batch_size = 10
epochs = 5

#model archtitecture:

model = Sequential()
model.add(Conv2D(16, kernel_size=(7, 7),
                 activation='relu', padding='same',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (7, 7), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (7, 7), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#choose as a loss function sparse_categorical_crossentropy so we dont have to do one-hot encoding
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#get a summary and make sure we dont have to many trainable parameters
print(model.summary())

#load the training samples (make sure both classes are present)
X = np.load('images.npy')[0:1000]
y = np.load('labels.npy')[0:1000]
print(X)
print(y)

#make a trainig test split or...
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

#...use the whole data to train
X_train = X
y_train = y

#free memory
del(y,X)

#train the model
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          #validation_data=(X_test, y_test)) #dont use if there is no test data
          

#save the model
model.save('model.h5')
print("Saved model to disk")
          
#evalute the model on the test set          
#score = model.evaluate(X_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
          
#load model
model = load_model('model.h5')
#summarize model.
model.summary()
          
#load test data (images and their labels) and make sure they are not the same as the once we used for training!!!
X = np.load('images_Muppets-03-04-03.npy')[2050:2250]
y = np.load('labels_Muppets-03-04-03.npy')[2050:2250]
#evaluate the model on the loaded data
score = model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
