import csv
import os
import scipy.ndimage
import numpy as np
import random
from sklearn.utils import shuffle
import cv2
import sklearn

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D

samples = []
f='../data/'
offset = 0.22
with open(f + 'driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        angle = float(row[3])
        if angle != 0.0 or random.random() < 0.10: # 10 percent of 'drive straight' images
            center = f + row[0].strip()
            sample = [center, False, angle]
            samples.append(sample)

            center = f + row[0].strip()
            sample = [center, True, -angle]
            samples.append(sample)

            left = f + row[1].strip()
            sample = [left, False, angle + offset]
            samples.append(sample)

            right = f + row[2].strip()
            sample = [center, False, angle - offset]
            samples.append(sample)

print('number of samples: {}'.format(len(samples)))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.05)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = scipy.ndimage.imread(batch_sample[0], mode='RGB')
                if batch_sample[1]:
                    image = np.fliplr(image)
                images.append(image)
                angles.append(batch_sample[2])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

def create_model():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3), output_shape=(160, 320, 3)))

    model.add(Convolution2D(6, 5, 5, subsample=(4, 4), border_mode='same'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(12, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(40, 3, 3, subsample=(2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(80))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model

model = create_model()
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
    validation_data=validation_generator, \
    nb_val_samples=len(validation_samples), nb_epoch=10)

model.save("./model.h5")
