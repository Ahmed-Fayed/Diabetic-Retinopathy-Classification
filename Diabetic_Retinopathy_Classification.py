# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 20:29:25 2021

@author: ahmed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import random
import os
import datetime
from tqdm import tqdm

import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0




# Reading data paths

resized_train_path = "E:/Software/professional practice projects/In progress/resized_train/resized_train"
resized_train_cropped_path = "E:/Software/professional practice projects/In progress/resized_train_cropped/resized_train_cropped"

train_labels_path = "E:/Software/professional practice projects/In progress/trainLabels.csv"
train_labels_cropped_path = "E:/Software/professional practice projects/In progress/trainLabels_cropped.csv"


# Exploring csv files

train_labels = pd.read_csv(train_labels_path)
print("train_labels head: ", train_labels.head())

print("train_labels info: ", train_labels.info())

level_column = train_labels['level']
print("level_column: ", level_column)

# level_column.hist(figsize=(10, 5))
# plt.suptitle("level_column")


level_column.plot(kind='hist', figsize=(10, 5), colormap = cm.get_cmap('flag'), title='level column')

train_labels_cropped = pd.read_csv(train_labels_cropped_path)
print("train_labels_cropped: ", train_labels_cropped)

print("train_labels_cropped info: ", train_labels_cropped.info())


level_cropped_col = train_labels_cropped['level']
print("level cropped col: ", level_cropped_col)

# level_cropped_col.hist(figsize=(10, 5))
level_cropped_col.plot(kind='hist', figsize=(10, 5), colormap=cm.get_cmap('ocean') , title="level_cropped_col")



# Visualizing resized train images

resized_train_list = os.listdir(resized_train_path)
print("resized train list: ", len(resized_train_list))



plt.figure(figsize=(25, 20))
plt.suptitle("random 5 by 5 sample images from resized train dataset")
for i in range(1, 26):
    plt.subplot(5, 5, i)
    img_name = random.choice(resized_train_list)
    img_path = os.path.join(resized_train_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.xlabel(img.shape[1])
    plt.ylabel(img.shape[0])

plt.show()


# Visualizing resized train cropped images

resized_train_cropped_list = os.listdir(resized_train_cropped_path)
print("resized train cropped list:", len(resized_train_cropped_list))



plt.figure(figsize=(26, 24))
plt.suptitle("random 5 by 5 sample imagesfrom resized train cropped dataset")
for i in range(1, 26):
    plt.subplot(5, 5, i)
    plt.tight_layout()
    # plt.title("random 5 by 5 sample images")
    img_name = random.choice(resized_train_cropped_list)
    img_path = os.path.join(resized_train_cropped_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.xlabel(img.shape[1])
    plt.ylabel(img.shape[0])

plt.show()

# turns out that the cropped images removes unimportant black backgraound so we we'll work on the cropped images dataset




# Image Processing

# we will perform two type of filtering
# First: Histogram Equalization
# Second: Ben Graham's Processing Method



# Explore histogram of a random image

plt.figure(figsize=(20, 25))
plt.suptitle("Color Histogram")
for i in range(1, 16):
    plt.subplot(5, 3, i)
    plt.tight_layout()
    # plt.title("Color Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Number of Pixels")
    img_name = random.choice(resized_train_cropped_list)
    img_path = os.path.join(resized_train_cropped_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    channels = cv2.split(img)
    colors = ['r', 'g', 'b']
    
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

plt.show()





######## Preprocessing Functions ##############

img_width = 100
img_height = 100


def read_img(img_name, resize=False):
    img_path = os.path.join(resized_train_cropped_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if resize:
        img = cv2.resize(img, (img_width, img_height))
    
    return img


def ben_graham(img):
    img_ben = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    return img_ben


def hist_equalization(img):
    red, green, blue = cv2.split(img)
    hist_red = cv2.equalizeHist(red)
    hist_green = cv2.equalizeHist(green)
    hist_blue = cv2.equalizeHist(blue)
    
    img_eq = cv2.merge((hist_red, hist_green, hist_blue))
    
    return img_eq





# Now we'll calculate histogram equility for some images with its label
# Taking a copy from the cropped images dataset(list) to apply histogram equalization

equal_hist_images = resized_train_cropped_list.copy()
print("equal hist images: ", len(equal_hist_images))


# Histogram Equalization Method

plt.figure(figsize=(26, 24))
plt.suptitle("Applying the Histogram Equaliztion")
counter = 0
for img_name in equal_hist_images:
    counter += 1
    plt.subplot(5, 5, counter)
    plt.tight_layout()
    # level_cropped_col is the labels, we've created it above
    plt.title(level_cropped_col[counter - 1])
    
    img = read_img(img_name)
    
    # Applying the Histogram Equaliztion
    img_eq = hist_equalization(img)
    
    plt.imshow(img_eq)
    plt.xlabel(img_eq.shape[1])
    plt.xlabel(img_eq.shape[0])
    
    if counter == 25:
        break

plt.show()



plt.figure(figsize=(20, 25))
plt.suptitle("Color Histogram the Histogram Equaliztion Method")
counter = 0
for img_name in equal_hist_images:
    counter += 1
    plt.subplot(5, 3, counter)
    plt.tight_layout()
    
    img = read_img(img_name)
    
    # Applying the Histogram Equaliztion
    img_eq = hist_equalization(img)
    
    channels = cv2.split(img_eq)
    colors = ['r', 'g', 'b']
    
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    
    if counter == 15:
        break

plt.show()



# Now we'll calculate Ben Graham's processing method for some images with its label
# Taking a copy from the cropped images dataset(list) to apply Ben Graham's processing method

# ben_images = resized_train_cropped_list.copy()
ben_images =  os.listdir(resized_train_cropped_path)
print("ben images: ", len(ben_images))


plt.figure(figsize=(26, 24))
plt.suptitle("Applying Ben Graham's Method")
counter = 0
for img_name in ben_images:
    counter += 1
    plt.subplot(5, 5, counter)
    plt.tight_layout()
    # level_cropped_col is the lebels list
    plt.title(level_cropped_col[counter - 1])
    
    img = read_img(img_name)
    
    # Applying Ben Graham's Method
    img_ben = ben_graham(img)
    
    plt.imshow(img_ben)
    plt.xlabel(img_ben.shape[1])
    plt.ylabel(img_ben.shape[0])
    
    if counter == 25:
        break

plt.show()




plt.figure(figsize=(20, 25))
plt.suptitle("Color Histogram for the Ben Graham's Method")
counter = 0
for img_name in equal_hist_images:
    counter += 1
    plt.subplot(5, 3, counter)
    plt.tight_layout()
    
    img = read_img(img_name)
    
    # Applying Ben Graham's Method
    img_ben = ben_graham(img)
    
    channels = cv2.split(img_ben)
    colors = ['r', 'g', 'b']
    
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    
    if counter == 15:
        break

plt.show()




# Figure out how the image will look like when resized to 100 by 100
# Because we will use the model EfficientNetB0 which expect input shape 100 by 100
img3 = cv2.resize(img, (100, 100))
plt.imshow(img3)
plt.show()


# Loading Data
# We'll Resize the data to be 100 by 100 to be valid input for EfficientNetB0

######### Different from my kaggle notebook ###########

images_list = []

for img_name in tqdm(resized_train_cropped_list):
    img = read_img(img_name, resize=True)
    
    images_list.append(img)


# images_list = []

# for img_name in tqdm(ben_images):
#     img = read_img(img_name, resize=True)
    
#     # Applying Ben Graham's Method
#     ben_img = ben_graham(img)
    
#     images_list.append(img_ben)




# images_list = []

# for img_name in tqdm(equal_hist_images):
#     img = read_img(img_name, resize=True)
    
#     # Applying the Histogram Equaliztion
#     img_eq = hist_equalization(img)
    
#     images_list.append(img_eq)





print("images_list: ", len(images_list))




# Making Augmentation

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                  rotation_range = 40,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  horizontal_flip = True,
                                  preprocessing_function=ben_graham)



val_datagen = ImageDataGenerator(rescale = 1./255.)



x_train, x_val, y_train, y_val = train_test_split(images_list, level_cropped_col, test_size=0.2, shuffle=True)

x_train = np.array(x_train)
x_val = np.array(x_val)

print('x_train: ', len(x_train))
print('x_val: ', len(x_val))
print('y_train: ', len(y_train))
print('y_val: ', len(y_val))



# Applying augmentation to datasets

train_datagen.fit(x_train)
val_datagen.fit(x_val)



# Create Model

# model = EfficientNetB0(include_top = False, weights='imagenet', input_shape=(100, 100, 3))

from tensorflow.keras.applications import MobileNet
model = MobileNet(input_shape=(100, 100, 3), include_top=False, weights='imagenet')
# Freeze pre-trained weights
model.trainable = False

print("model trainable weights: ", model.trainable_weights)


x = Dropout(0.2)(model.output)

x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.1)(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.1)(x)

x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.1)(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.1)(x)


x = GlobalAveragePooling2D()(x)
x = Dropout(0.1)(x)
classifier = Dense(5, activation='softmax')(x)

model = Model(inputs=model.input, outputs=classifier)


model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['SparseCategoricalAccuracy'])


model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_val, y_val), verbose=1,
          callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
                    CSVLogger("train.csv")])




model.save("Diabetic_Retinopathy_BenGrahamData.h5")
model.save_weights("Diabetic_Retinopathy_Weights_BenGrahamData.h5")

json_model = model.to_json()
with open("E:/Software/professional practice projects/In progress/Diabetic_Retinopathy_BenGrahamData.json", 'w') as json_file:
    json_file.write(json_model)




def plot_accuray(history):
    plt.figure(figsize=(12, 8))
    plt.title("Diabetic_Retinopathy_BenGrahamModel Accuracy")
    plt.plot(history.history['sparse_categorical_accuracy'], color='g')
    plt.plot(history.history['val_sparse_categorical_accuracy'], color='b')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['train', 'val'], loc="lower right")
    plt.show()

def plot_loss(history):
    plt.figure(figsize=(12, 8))
    plt.title("Diabetic_Retinopathy_BenGraham Model Loss")
    plt.plot(history.history['loss'], color='g')
    plt.plot(history.history['val_loss'], color='b')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['train', 'val'], loc="lower right")
    plt.show()



history = model.history

plot_accuray(history)
plot_loss(history)