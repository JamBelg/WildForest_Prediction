# import library
import numpy as np
import pandas as pd
import os

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping

# Plot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageFile, Image

# Sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.applications import VGG16

train_dir = 'data/train'
test_dir = 'data/test'
valid_dir = 'data/valid'

rescale_datagen = ImageDataGenerator(
    dtype='float32',                # Data type for the output data
    rescale=1./255.,                # Rescale pixel values to the range [0, 1]
    rotation_range=10,              # Randomly rotate images by up to 10 degrees
    zoom_range=0.05,                # Randomly zoom in/out on images by 5%
    width_shift_range=0.1,          # Randomly shift the width of images by 10%
    height_shift_range=0.1,         # Randomly shift the height of images by 10%
    shear_range=0.15,               # Randomly apply shear transformations
    horizontal_flip=True,           # Randomly flip images horizontally
    fill_mode="nearest"             # Strategy for filling in newly created pixels
)
train_generator = rescale_datagen.flow_from_directory(train_dir, 
                                                      batch_size = 50, 
                                                      target_size = (250,250),
                                                      color_mode = "rgb",
                                                      class_mode = "categorical",
                                                      shuffle = True,
                                                      seed = 42)
valid_generator = rescale_datagen.flow_from_directory(valid_dir,
                                                      batch_size = 50,
                                                      target_size = (250,250),
                                                      color_mode = "rgb",
                                                      class_mode = "categorical",
                                                      shuffle = True,
                                                      seed = 42)
test_generator = rescale_datagen.flow_from_directory(test_dir,
                                                     batch_size = 50,
                                                     target_size = (250,250),
                                                     color_mode = "rgb",
                                                     class_mode = "categorical",
                                                     shuffle = False,
                                                     seed = 42)


vgg = VGG16(weights='imagenet', include_top=False, input_shape=(250,250,3))
for layer in vgg.layers:
    layer.trainable=False

model3 = Sequential([
    vgg,
    Flatten(),
    Dense(units=512, activation='elu'),
    Dense(units=128, activation='elu'),
    Dense(units=2, activation='softmax')
])

model3.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

callback = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

logs3 = model3.fit(train_generator,
                   epochs = 20,
                   steps_per_epoch=30250/50,
                   validation_data = valid_generator,
                   validation_steps=6300/50,
                   callbacks=[callback])

model3.save('Model_vgg.h5')

model3.evaluate(test_generator, steps=6300/50)