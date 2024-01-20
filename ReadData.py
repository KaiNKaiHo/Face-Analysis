import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
import keras.applications.vgg16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
import seaborn as sns
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.utils import resample


class ReadData:
    def __init__(self, file_path, img_directory):
        self.df = pd.read_csv(file_path)
        self.df = self.df.astype(str)
        self.img_directory = img_directory

    def data_preprocessing(self, X_train, y_train, X_test, y_test, batch_size=64):

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            preprocessing_function=keras.applications.vgg16.preprocess_input,
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(
            pd.DataFrame({'paths': X_train,
                          'age': y_train[:, 0],
                          'race': y_train[:, 1],
                          'skintone': y_train[:, 2],
                          'emotion': y_train[:, 3],
                          'gender': y_train[:, 4]
                          }),
            x_col='paths',
            y_col=['age', 'race', 'skintone', 'emotion', 'gender'],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='multi_output',
            seed=42,
            shuffle=True
        )

        test_generator = test_datagen.flow_from_dataframe(
            pd.DataFrame({'paths': X_test,
                          'age': y_test[:, 0],
                          'race': y_test[:, 1],
                          'skintone': y_test[:, 2],
                          'emotion': y_test[:, 3],
                          'gender': y_test[:, 4]
                          }),
            x_col='paths',
            y_col=['age', 'race', 'skintone', 'emotion', 'gender'],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='multi_output',
            seed=42,
            shuffle=True
        )

        return train_generator, test_generator

    def preprocess_data(self, df):
        # Lấy đường dẫn đầy đủ đến ảnh
        image = np.array(df['array_images'])
        labels = list()
        tasks = ['age', 'race', 'masked', 'skintone', 'emotion', 'gender']
        for task in tasks:
            label = []
            if task == 'age':
                label = df['age'].astype(str).map({'20-30s': 0,
                                                   '40-50s': 1,
                                                   'Kid': 2,
                                                   'Senior': 3,
                                                   'Teenager': 4,
                                                   'Baby': 5})
            elif task == 'masked':
                # Lấy nhãn 'masked' và chuyển đổi sang chuỗi
                label = df['masked'].astype(str).map({'unmasked': 0, 'masked': 1})

            elif task == 'skintone':
                label = df['skintone'].astype(str).map({'mid-light': 0,
                                                        'light': 1,
                                                        'mid-dark': 2,
                                                        'dark': 3})

            elif task == 'emotion':
                label = df['emotion'].astype(str).map({'Happiness': 0,
                                                       'Neutral': 1,
                                                       'Sadness': 2,
                                                       'Anger': 3,
                                                       'Surprise': 4,
                                                       'Disgust': 5,
                                                       'Fear': 6})

            elif task == 'race':
                label = df['race'].astype(str).map({'Mongoloid': 0,
                                                    'Caucasian': 1,
                                                    'Negroid': 2})

            elif task == 'gender':
                label = df['gender'].astype(str).map({'Female': 0,
                                                      'Male': 1})
            label = np.array(label)
            labels.append(label)
        final_label = np.concatenate([labels[0][:, np.newaxis],
                                      labels[1][:, np.newaxis],
                                      labels[2][:, np.newaxis],
                                      labels[3][:, np.newaxis],
                                      labels[4][:, np.newaxis],
                                      labels[5][:, np.newaxis]], axis=1)

        # Chia dữ liệu thành tập train và test
        X_train, X_test, y_train, y_test = train_test_split(image, final_label, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test



   
