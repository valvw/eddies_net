import os
import numpy as np
import matplotlib.pyplot as plt


import glob
import xarray as xr
import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import *

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tqdm.auto import tqdm
import re


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from tensorflow.keras import mixed_precision

# Enable mixed precision with Tensor Core operations
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
global_dtype_policy = tf.keras.mixed_precision.global_policy().name



def sort_by_last_two_elements(file_path):
    filename = os.path.basename(file_path)
    
    # Use regular expression to find all numeric parts in the filename
    numeric_parts = [int(part) for part in re.findall(r'\d+', filename)]
    
    if numeric_parts:
        # Combine the constant part and all numeric parts
        constant_part = re.sub(r'\d+', '', filename)  # Remove numeric parts
        return (constant_part, numeric_parts)
    else:
        return filename


dir_i = r'data\patches_i'
dir_m = r'data\patches_m'
images = sorted(glob.glob(os.path.join(dir_i, "*.tif")))
#masks = sorted(glob.glob(os.path.join(dir_m, "*.tif")))

train_size = int(len(images)*0.8)



def load_with_generator(dir_i, dir_m):
    def gen():
        images = sorted((glob.glob(os.path.join(dir_i, "*.tif")))[:train_size], key=sort_by_last_two_elements)
        masks = sorted((glob.glob(os.path.join(dir_m, "*.tif")))[:train_size], key=sort_by_last_two_elements)
        for image, mask in zip(images, masks):
        #for image, mask in zip((glob.glob(os.path.join(dir_i, "*.tif"))), (glob.glob(os.path.join(dir_m, "*.tif")))):
            ds = xr.open_dataset(image).band_data.values
            #input_data = tf.convert_to_tensor(ds)
            input_data = tf.expand_dims(tf.convert_to_tensor(ds), axis=-1)
            input_data = tf.cast(input_data, tf.float32)
            
            # Load the corresponding mask (modify the path accordingly)
            mask_ds = xr.open_dataset(mask, decode_cf=False).band_data.values
            #mask_data = tf.convert_to_tensor(mask_ds)
            mask_data = tf.expand_dims(tf.convert_to_tensor(mask_ds), axis=-1)
            mask_data = tf.cast(mask_data, tf.int8)
            
            yield input_data[0], mask_data[0]  # Remove the singleton batch dimension


    sample = next(iter(gen()))

    # Create a dataset from the generator
    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.int8),
        output_shapes=(
            tf.TensorShape([512, 512, 1]),
            tf.TensorShape([512, 512, 1])
        )
    )

num_classes = 3


dataset = load_with_generator(dir_i, dir_m)
dataset

def unet_model(input_shape, num_classes):
    # U-Net architecture
    inputs = tf.keras.Input(shape=input_shape)

    # Contracting path (down-sampling)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)  
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1) 
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2) 
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)  
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Middle part
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)  
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)  

    # Expanding path (up-sampling)
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    concat1 = layers.concatenate([conv2, up1], axis=-1)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat1)
    conv4 = layers.BatchNormalization()(conv4)  
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)  

    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    concat2 = layers.concatenate([conv1, up2], axis=-1)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat2)
    conv5 = layers.BatchNormalization()(conv5)  
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)  

    # Output layer with sigmoid activation based on dtype policy
    if global_dtype_policy in ["mixed_float16", "mixed_bfloat16"]:
        outputs = layers.Conv2D(num_classes, 1, activation=tf.keras.activations.get('sigmoid'), dtype=tf.float32)(conv5)
    else:
        outputs = layers.Conv2D(num_classes, 1, activation=tf.keras.activations.get('sigmoid'))(conv5)


    # Output layer
    #outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Assuming input shape of (512, 512, 1) for grayscale images
input_shape = (512, 512, 1)

# Create the U-Net model
model = unet_model(input_shape, num_classes)

# Print the model summary
model.summary()


from keras.optimizers import Adam

# Define hyperparameters
batch_size = 2
num_epochs = 5
optimizer = Adam(learning_rate=3e-4)
loss_function = 'sparse_categorical_crossentropy'
activation_function = 'relu'
scaled_optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic = True)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model_checkpoint = ModelCheckpoint('model.h5', verbose = 1, save_best_only=True, 
                                   save_weights_only=True)

class updated_meanIoU(tf.keras.metrics.MeanIoU):
    
    def __init__(self, y_true=None, y_pred=None, num_classes=None, name=None, dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes = num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        
        return super().update_state(y_true, y_pred, sample_weight)
    
metrics = [
    'accuracy',
    updated_meanIoU(num_classes = num_classes)
]

model.compile(optimizer=scaled_optimizer, loss = loss_function, metrics = metrics)

history = model.fit(dataset.batch(batch_size), epochs=num_epochs, callbacks=[model_checkpoint])

model.save("eddies_20_epochs.hdf5")

