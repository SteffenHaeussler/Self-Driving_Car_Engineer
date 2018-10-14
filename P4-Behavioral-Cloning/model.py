import numpy as np
from pandas import read_csv

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import Flatten, Dense, Cropping2D, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import Adam

from scipy import ndimage
import sklearn.utils as utils
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



def read_data(filename):

    # i don't want to load the whole pandas library for reading a csv; but pandas is the most comfortable here
    _COLUMNS = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
    data = read_csv(filename, delimiter=',',header=None, names=_COLUMNS)

    return data


def generator(samples, batch_size):

    num_samples = len(samples)
    while True: # generator never stops
        samples = utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []

            # in case, i want to use more than the steering angle
            measurement = batch_samples[['steering_angle', 'throttle', 'break', 'speed']]
            image_path = batch_samples['center_image']

            # add a binary variable for vertical flipping of images; here 70 - 30 ratio
            flip = np.random.choice([True, False], measurement.shape[0], p=[0.3, 0.7])

            # flipping the steering angle
            measurement['steering_angle'] = [val * -1 if flip else val
                                             for val, flip in zip(measurement['steering_angle'], flip)]

            for img, flip in zip(image_path, flip):

                center_image = ndimage.imread(img)
                # vertical flipping of the choosen images
                if flip:
                    center_image = np.flip(center_image, 1)

                images.append(center_image[:,:,:])

            # get X and y as numpy arrays
            X_train = np.array(images)
            y_train = measurement['steering_angle'].values

            yield utils.shuffle(X_train, y_train)


def model_train(data, filepath, learning_rate, epochs, batch_size=32):

    # train validation split
    train_samples, validation_samples = train_test_split(data, test_size=0.2)

    # generator runs for training and validation data
    train_generator = generator(train_samples, batch_size)
    validation_generator = generator(validation_samples, batch_size)

    #nvidia architecture
    # https://arxiv.org/pdf/1604.07316v1.pdf
    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
    # model.add(Lambda(lambda x: x / 255.0))
    # model.add(Lambda(lambda x: x[:,:,::2,:]))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=24,kernel_size=5, strides=2, padding='valid', activation='relu'))
    model.add(Conv2D(filters=36,kernel_size=5, strides=2, padding='valid', activation='relu'))
    model.add(Conv2D(filters=48,kernel_size=5, strides=2, padding='valid', activation='relu'))
    model.add(Conv2D(filters=64,kernel_size=3, strides=2, padding='valid', activation='relu'))
    model.add(Conv2D(filters=64,kernel_size=3, strides=1, padding='valid', activation='relu'))

    model.add(Flatten())

    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    print(model.summary())

    # saving best model; should prevent overfitting
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # early stopping, should stop training, when no improvement happens
    early_stop = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=2,
                                  verbose=0, mode='auto')
    #Default parameters follow those provided in the original paper.
    opt = Adam(lr=learning_rate)

    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    # run the keras model with the train & validation generators
    history = model.fit_generator(train_generator,
                                 steps_per_epoch=len(train_samples)//batch_size,
                                 validation_data=validation_generator,
                                 validation_steps=len(validation_samples)//batch_size,
                                 epochs=epochs,
                                 callbacks=[early_stop, checkpoint],
                                 verbose=1)

    return history


# plotting of training and validation loss
def plot_model_output(history):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('simple_model_training_loss.png')


def main():

    input_file = 'train_data/driving_log.csv'
    output_file = "model.h5"

    batch_size = 128
    epochs = 20
    learning_rate = 0.001

    data = read_data(input_file)
    history = model_train(data, output_file, learning_rate, epochs, batch_size)
    plot_model_output(history)


if __name__ == "__main__":
    main()


