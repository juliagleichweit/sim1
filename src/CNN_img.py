import os.path as osp
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, Input
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K
import numpy as np

should_train = False
should_test = True

# path to data images
train_dir = "../data/train"
test_dir = "../data/test"

nb_train_samples = 3088
nb_validation_samples = 1549

# image dimensions
img_width = 720
img_height = 544

classes = 2
batch_size = 16
epochs = 2

# paths to model weights
top_weights_path = "../data/model/best_weights_sigmoid.h5"
top_model_path = "../data/model/best_model_sigmoid.h5"


def load_train_data():
    """
    Load training images from train_dir and split data into
    80% training and 20% validation
    :return:
    """
    data_generator = ImageDataGenerator(rescale=1. / 255,
                                        validation_split=0.2)  # split our train data in 80% training and 20% validation data
    train_generator = data_generator.flow_from_directory(directory=train_dir,
                                                         target_size=(img_height, img_width),
                                                         class_mode='binary',  # since we only have 2 classes
                                                         batch_size=batch_size,
                                                         subset='training')

    validation_generator = data_generator.flow_from_directory(directory=train_dir,
                                                              target_size=(img_height, img_width),
                                                              class_mode='binary',
                                                              batch_size=batch_size,
                                                              subset='validation')
    return train_generator, validation_generator


def prepare_model():
    print("prepare model")
    # model archtitecture:
    print("create model")
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, 3)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    from keras.optimizers import SGD
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9)

    # from keras.optimizers import Adam
    # opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    print("compile model")
    model.summary()

    return model


if __name__ == '__main__':

    if should_train:
        print("training mode")
        train_gen, val_den = load_train_data()

        if osp.exists(top_model_path): # continue from the last training
            from keras.models import load_model

            model = load_model(top_model_path)
            # model.load_weights(top_weights_path)
            print("model weights loaded")
            # complete model is loaded including architecture, weights, optimizer
        else:
            model = prepare_model()

        # use callbacks EarlyStopping and ModelCheckpoint
        from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

        csv_logger = CSVLogger('training.log', append=True)
        checkpoint = ModelCheckpoint(top_model_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
        # early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=1, mode='auto')

        # Fitting/Training the model
        print("training the model")
        model.fit_generator(train_gen, steps_per_epoch=train_gen.samples // batch_size, epochs=epochs,
                            validation_data=val_den,
                            validation_steps=val_den.samples // batch_size,
                            callbacks=[csv_logger, checkpoint])

    if should_test:
        if osp.exists(top_model_path):
            if not should_train: # if we previously trained we do not load
                from keras.models import load_model
                model = load_model(top_model_path)
                # model.load_weights(top_weights_path)
                print("model loaded")
            else:
                model = prepare_model()

            test_datagen = ImageDataGenerator(rescale=1. / 255)
            validation_generator = test_datagen.flow_from_directory(
                directory=test_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='binary', shuffle=False)

            print("evaluating model")
            score = model.evaluate_generator(validation_generator, nb_validation_samples / batch_size)

            # validation_generator.reset()
            # scores = model.predict_generator(validation_generator, nb_validation_samples / batch_size)

            # class_one = scores > 0.5
            # true_labels = np.array([0] * 492 + [1] * 1057)
            # acc = np.mean(class_one == true_labels)
            # print("acc: ", acc)

            # correct = 0
            # for i, n in enumerate(validation_generator.filenames):
            #   print(n ," score: ", scores[i][0], (scores[i][0]<=0.5))

            print("Loss: ", score[0], "Accuracy: ", score[1])
        else:
            print("cannot find trained model")
