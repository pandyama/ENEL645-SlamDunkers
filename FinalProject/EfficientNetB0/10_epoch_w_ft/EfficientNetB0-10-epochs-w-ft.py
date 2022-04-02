#!/usr/bin/env python
# coding: utf-8

import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pylab as plt
import pickle

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Defining data generators and data augmentation parameters
gen_params = {"featurewise_center":False,\
              "samplewise_center":False,\
              "featurewise_std_normalization":False, \
              "samplewise_std_normalization":False,\
              "zca_whitening":False, \
              "rotation_range":20, \
               "width_shift_range":0.1,   \
               "height_shift_range":0.1,  \
               "shear_range":0.2,            \
               "zoom_range":0.1,    \
               "horizontal_flip":True,   \
               "vertical_flip":True}

train_gen = ImageDataGenerator(**gen_params, preprocessing_function = tf.keras.applications.efficientnet.preprocess_input)
val_gen = ImageDataGenerator(**gen_params, preprocessing_function = tf.keras.applications.efficientnet.preprocess_input)

class_names = ["Black", "Blue",  "Green", "Take-to-recycle"]

bs = 64 # batch size

train_generator = train_gen.flow_from_directory(
    directory = "/work/TALC/enel645_2022w/Garbage-dataset/Train",
    target_size=(256, 256),
    color_mode="rgb",
    classes= class_names,
    class_mode="categorical",
    batch_size=bs,
    shuffle=True,
    seed=42,
    interpolation="nearest",
)

validation_generator = val_gen.flow_from_directory(
    directory = "/work/TALC/enel645_2022w/Garbage-dataset/Validation",
    target_size=(256, 256),
    color_mode="rgb",
    classes= class_names,
    class_mode="categorical",
    batch_size=bs,
    shuffle=True,
    seed=42,
    interpolation="nearest",)

# Defining callbacks
model_name_it = "/home/makissin/TL-garbage-classification/Outputs/garbage_classifier_en_b0_10_epochs_it.h5"
model_name_ft = "/home/makissin/TL-garbage-classification/Outputs/garbage_classifier_en_b0_10_epochs_ft.h5"

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10)

monitor_it = tf.keras.callbacks.ModelCheckpoint(model_name_it, monitor='val_loss',                                             verbose=0,save_best_only=True,                                             save_weights_only=False,                                             mode='min')

monitor_ft = tf.keras.callbacks.ModelCheckpoint(model_name_ft, monitor='val_loss',                                             verbose=0,save_best_only=True,                                             save_weights_only=False,                                             mode='min')

def scheduler(epoch, lr):
    if epoch%10 == 0 and epoch!= 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

img_height = 256
img_width = 256

 # Defining the model
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(img_height, img_width, 3),
    include_top=False
)
base_model.trainable = False #Set to true to ignore

#without dropout layer
x1 = base_model(base_model.input, training = False)
x2 = tf.keras.layers.Flatten()(x1)

#with drop out layer
# x1 = base_model(base_model.input, training = False)
# x2 = keras.layers.Dropout(0.2)(x1)
# x3 = tf.keras.layers.Flatten()(x2)

out = tf.keras.layers.Dense(len(class_names),activation = 'softmax')(x2)
model = tf.keras.Model(inputs = base_model.input, outputs =out)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Initial training - just the top (classifier)
history_it = model.fit(train_generator, epochs=10, verbose = 1,callbacks= [early_stop, monitor_it, lr_schedule],\
                       validation_data = (validation_generator))


# Saving the training history
it_file = open("it_history.pkl", "wb")
pickle.dump(history_it.history, it_file)
it_file.close()

# Fine-tuning the model
model = tf.keras.models.load_model(model_name_it)
model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-8),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history_ft = model.fit(train_generator, epochs=5, verbose = 1, \
                       callbacks= [early_stop, monitor_ft, lr_schedule],  \
                       validation_data = (validation_generator))


# Saving the fine-tuning history
ft_file = open("ft_history.pkl", "wb")
pickle.dump(history_ft.history, ft_file)
it_file.close()

# Saving the plot to a file
acc_it = history_it.history['accuracy']
val_acc_it = history_it.history['val_accuracy']
loss_it = history_it.history['loss']
val_loss_it = history_it.history['val_loss']
epochs_range = range(2)
plt.figure()
plt.plot(acc_it, label='Training Accuracy')
plt.plot(val_acc_it, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
plt.savefig('Accuracy_IT_10_epochs.png')
plt.plot(loss_it, label='Training Loss')
plt.plot(val_loss_it, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig('Loss_IT.png')

acc_ft = history_ft.history['accuracy']
val_acc_ft = history_ft.history['val_accuracy']
loss_ft = history_ft.history['loss']
val_loss_ft = history_ft.history['val_loss']
epochs_range = range(2)
plt.figure()
plt.plot(acc_ft, label='Training Accuracy')
plt.plot(val_acc_ft, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
plt.savefig('Accuracy_FT_10_epochs.png')
plt.plot(loss_ft, label='Training Loss')
plt.plot(val_loss_ft, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig('Loss_FT_10_epochs.png')