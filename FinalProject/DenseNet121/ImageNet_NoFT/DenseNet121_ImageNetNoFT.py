#!/usr/bin/env python
# coding: utf-8

import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from sklearn.metrics import confusion_matrix, classification_report
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


train_gen = ImageDataGenerator(**gen_params, preprocessing_function = tf.keras.applications.densenet.preprocess_input)
val_gen = ImageDataGenerator(**gen_params, preprocessing_function = tf.keras.applications.densenet.preprocess_input)
test_gen = ImageDataGenerator(**gen_params, preprocessing_function = tf.keras.applications.densenet.preprocess_input)

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

test_generator = test_gen.flow_from_directory(
    directory = "/work/TALC/enel645_2022w/Garbage-dataset/Test",
    target_size=(256, 256),
    color_mode="rgb",
    classes= class_names,
    class_mode="categorical",
    batch_size=bs,
    shuffle=True,
    seed=42,
    interpolation="nearest",)


# Defining callbacks

model_name_it = "/home/meet.pandya/Test-TL/Outputs/garbage_classifier_en_b0_it.h5"
model_name_ft = "/home/meet.pandya/Test-TL/Outputs/garbage_classifier_en_b0_ft.h5"

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 10)
monitor_it = tf.keras.callbacks.ModelCheckpoint(model_name_it, monitor='val_loss',                                             verbose=0,save_best_only=True,                                             save_weights_only=False,                                             mode='min')

monitor_ft = tf.keras.callbacks.ModelCheckpoint(model_name_ft, monitor='val_loss',                                             verbose=0,save_best_only=True,                                             save_weights_only=False,                                             mode='min')

def scheduler(epoch, lr):
    if epoch%10 == 0 and epoch!= 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)



# Defining the model


img_height = 256
img_width = 256

# Defining the model
base_model = tf.keras.applications.DenseNet121(
    weights='imagenet',
    input_shape=(img_height, img_width, 3),
    include_top=False)
base_model.trainable = False

x1 = base_model(base_model.input, training = False)
x2 = tf.keras.layers.Flatten()(x1)


out = tf.keras.layers.Dense(len(class_names),activation = 'softmax')(x2)
model = tf.keras.Model(inputs = base_model.input, outputs =out)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Initial training - just the top (classifier)
history_it = model.fit(train_generator, epochs=150, verbose = 1,callbacks= [early_stop, monitor_it, lr_schedule],validation_data = (validation_generator))


# Saving the training history
it_file = open("it_history.pkl", "wb")
pickle.dump(history_it.history, it_file)
it_file.close()

## Saving plot to a file

acc = history_it.history['accuracy']
val_acc = history_it.history['val_accuracy']

loss = history_it.history['loss']
val_loss = history_it.history['val_loss']

epochs_range = range(150)

plt.figure()
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
plt.savefig('AccuracyIT.png')

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig('LossIT.png')