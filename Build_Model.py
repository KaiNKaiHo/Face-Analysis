import tensorflow as tf
import keras.models
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout, Input, \
    Concatenate
from keras.models import Sequential, Model
from keras.applications import mobilenet_v2, vgg16, inception_resnet_v2
from keras.optimizers import Adam
import cv2
import numpy as np
import time


class BuildModelClassify:
    def __init__(self, base_model=None, input_shape=(224, 224, 3)):
        if base_model is None:
            self.base_model = vgg16.VGG16(weights='imagenet', include_top=False,
                                          input_shape=input_shape)
        else:
            self.base_model = base_model
        for layer in self.base_model.layers:
            layer.trainable = False

    def build(self):
        input = Input(shape=(224, 224, 3), name='input')
        base = self.base_model(input)
        base = GlobalAveragePooling2D()(base)
        base = Dense(1024, activation='relu', name='base')(base)
        base = Dropout(0.5)(base)
        # Task age
        task_age1 = Dense(256, activation='relu', name='Age1')(base)
        task_age2 = Dense(6, activation='softmax', name='Age')(task_age1)

        # # Freezing layer
        # task_age1.trainable = False

        # Task masked
        task_masked1 = Dense(256, activation='relu', name='Masked1')(base)
        task_masked2 = Dense(2, activation='softmax', name='Masked')(task_masked1)

        # Task race
        task_race1 = Dense(256, activation='relu', name='Race1')(base)
        task_race2 = Dense(3, activation='softmax', name='Race')(task_race1)

        # Task skintone
        task_skintone1 = Dense(256, activation='relu', name='Skintone1')(base)
        task_skintone2 = Dense(4, activation='softmax', name='Skintone')(task_skintone1)

        # Task gender
        task_gender1 = Dense(256, activation='relu', name='Gender1')(base)
        task_gender2 = Dense(2, activation='softmax', name='Gender')(task_gender1)

        # Task emotion
        task_emotion1 = Dense(256, activation='relu', name='Emotion1')(base)
        task_emotion2 = Dense(7, activation='softmax', name='Emotion')(task_emotion1)

        model = Model(inputs=input,
                      outputs=[task_age2, task_race2, task_masked2, task_skintone2, task_emotion2, task_gender2])

        for layer in self.base_model.layers:
            layer.trainable = False

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss={'Age': 'categorical_crossentropy',
                            'Race': 'categorical_crossentropy',
                            'Masked': 'categorical_crossentropy',
                            'Skintone': 'categorical_crossentropy',
                            'Emotion': 'categorical_crossentropy',
                            'Gender': 'categorical_crossentropy',
                            },
                      metrics=['accuracy'])
        return model

    def train(self, model, data_train, data_test, epochs=10, batch_size=32):
        model_history = model.fit(data_train,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  validation_data=data_test)
        return model_history
