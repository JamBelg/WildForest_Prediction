# import library
import numpy as np
import pandas as pd
import os

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, LSTM
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

rescale_datagen = ImageDataGenerator(dtype='float32',
                                     rescale= 1./255.,
                                     rotation_range=20,
                                     zoom_range=0.15,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.15,
                                     horizontal_flip=True,
                                     fill_mode="nearest")
train_generator = rescale_datagen.flow_from_directory(train_dir, 
                                                      batch_size = 50, 
                                                      target_size = (224,224),
                                                      color_mode = "rgb",
                                                      class_mode = "categorical",
                                                      shuffle = True,
                                                      seed = 42)
valid_generator = rescale_datagen.flow_from_directory(valid_dir,
                                                      batch_size = 50,
                                                      target_size = (224,224),
                                                      color_mode = "rgb",
                                                      class_mode = "categorical",
                                                      shuffle = True,
                                                      seed = 42)
test_generator = rescale_datagen.flow_from_directory(test_dir,
                                                     batch_size = 50,
                                                     target_size = (224,224),
                                                     color_mode = "rgb",
                                                     class_mode = "categorical",
                                                     shuffle = False,
                                                     seed = 42)

vgg16file = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
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

model3.save('Model3.h5')

model3.evaluate(test_generator, steps=6300/50)