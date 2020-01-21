import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Flatten, Dense, Dropout, Flatten, Conv1D, MaxPooling1D
import os.path as osp

# We build a CNN for the classification of the audio which comes as Mel-frequency cepstral coefficients

# specify hyper parameters
num_classes = 2
input_shape = (40,44)  # we have 40 Mel-frequency cepstral coefficients

batch_size = 1000
epochs = 3

should_train = True
should_test = True
save_probabilities = True

# path to data images
train_file = "../data/trainchunks.npy"
test_file = "../data/testchunks.npy"

nb_train_samples = 3088
nb_validation_samples = 1549

classes = 2
batch_size = 128
epochs = 10

# paths to model weights
top_weights_path = "../data/model/audio_best_weights_sigmoid.h5"
top_model_path = "../model_audio.h5"


def prepare_model():
    print("prepare model")
    # model archtitecture:
    # specify the model architecture

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='softmax'))

    #from keras.optimizers import SGD
    #opt = SGD(lr=0.01, decay=1e-6, momentum=0.9)

    from keras.optimizers import Adam
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    print("compile model")
    model.summary()

    return model


if __name__ == '__main__':

    if should_train:
        print("training mode")

        # load data
        X = np.load(train_file)  # [0:10000]
        # we have Muppets-02-01-01 kermit #82, not_kermit #1465
        #         Muppets-03-04-03 kermit #307, not_kermit #1233
        # in exactly this order
        y = np.asarray([0] * 82 + [1] * 1465 + [0] * 307 + [1] * 1233)
        print(X.shape ," \n", y.shape)
        # make a trainig test split or...
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

        if osp.exists(top_model_path):  # continue from the last training
            from keras.models import load_model

            model = load_model(top_model_path)
            # model.load_weights(top_weights_path)
            print("model weights loaded")
            # complete model is loaded including architecture, weights, optimizer
        else:
            model = prepare_model()

        # use callbacks EarlyStopping and ModelCheckpoint
        from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

        csv_logger = CSVLogger('audio_training.log', append=True)
        checkpoint = ModelCheckpoint(top_model_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
        # early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=1, mode='auto')
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_valid, y_valid),
                  callbacks=[csv_logger,checkpoint])  # dont use if there is no test data

        # Fitting/Training the model
        print("training the model")

    if should_test:
        if osp.exists(top_model_path):
            if not should_train:  # if we previously trained we do not load
                from keras.models import load_model

                model = load_model(top_model_path)
                # model.load_weights(top_weights_path)
                print("model loaded")
            else:
                model = prepare_model()

            # load data
            X_test = np.load(test_file)  # [0:10000]
            # we have Muppets-02-04-04 kermit #252, not_kermit #1296
            # in exactly this order
            y_test = np.asarray([0] * 252 + [1] * 1296)

            score = model.evaluate(X_test, y_test, batch_size=batch_size,verbose=0)

            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

            print("evaluating model")

            if save_probabilities:
                scores = model.predict(X_test, batch_size=batch_size)
                np.save("probabilites_test_audio", scores)
                #print(scores)

            print("Loss: ", score[0], "Accuracy: ", score[1])
        else:
            print("cannot find trained model")
