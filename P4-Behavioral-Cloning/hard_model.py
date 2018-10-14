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

    _COLUMNS = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
    data = read_csv(filename, delimiter=',',header=None, names=_COLUMNS)

    return data


def generator(samples, batch_size, correction):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        samples = utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering = []

            measurement = batch_samples[['steering_angle', 'throttle', 'break', 'speed']]
            image_path = batch_samples['center_image']
            left_image_path = batch_samples['left_image']
            right_image_path = batch_samples['right_image']

            flip = np.random.choice([True, False], measurement.shape[0], p=[0.3, 0.7])

            # measurement['steering_angle'] = [val * -1 if flip else val
            #                                  for val, flip in zip(measurement['steering_angle'], flip)]
            for i in range(measurement.shape[0]):

                center_image = ndimage.imread(image_path.values[i])
                left_image = ndimage.imread(left_image_path.values[i])
                right_image = ndimage.imread(right_image_path.values[i])

                center_steer = measurement['steering_angle'].values[i]
                left_steer = measurement['steering_angle'].values[i] + correction
                right_steer = measurement['steering_angle'].values[i] - correction

                if flip[i]:
                    center_image = np.flip(center_image, 1)
                    left_image = np.flip(left_image, 1)
                    right_image = np.flip(right_image, 1)

                    center_steer *= -1
                    left_steer *= -1
                    right_steer *= -1

                images.append(center_image[:,:,:])
                images.append(left_image[:,:,:])
                images.append(right_image[:,:,:])

                steering.append(center_steer)
                steering.append(left_steer)
                steering.append(right_steer)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(steering)

            yield utils.shuffle(X_train, y_train)



def model_train(data, filepath, learning_rate, epochs, batch_size=32, correction=0.2):


    train_samples, validation_samples = train_test_split(data, test_size=0.2)

    train_generator = generator(train_samples, batch_size, correction)
    validation_generator = generator(validation_samples, batch_size, correction)

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

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early_stop = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=2,
                                  verbose=0, mode='auto')
    #Default parameters follow those provided in the original paper.
    opt = Adam(lr=learning_rate)

    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    history = model.fit_generator(train_generator,
                                 steps_per_epoch=3*len(train_samples)//batch_size,
                                 validation_data=validation_generator,
                                 validation_steps=3*len(validation_samples)//batch_size,
                                 epochs=epochs,
                                 callbacks=[early_stop, checkpoint],
                                 verbose=1)

    return history


def plot_model_output(history):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('hard_model_training_loss.png')


def main():
    input_file = 'hard_train_data/driving_log.csv'
    output_file = "hard_model.h5"

    batch_size = 128
    epochs =20
    learning_rate = 0.001
    camera_correction = 0.2

    data = read_data(input_file)
    history = model_train(data, output_file, learning_rate, epochs, batch_size, camera_correction)
    plot_model_output(history)

if __name__ == "__main__":
    main()








