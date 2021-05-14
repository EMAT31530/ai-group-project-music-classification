#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 16:14:23 2021

@author: jayvier
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization,
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D,
                          GlobalMaxPooling2D, Dropout)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
import os
import random
from hyperopt import STATUS_OK
import uuid


data = '../spectrograms3secXX'
genres = list(os.listdir(f'{data}/train/'))
genres.remove('.DS_Store')
print(genres)

data_gen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

path=f'{data}/train' 
image_datagentrain=data_gen.flow_from_directory(path,target_size=(128,130),class_mode='categorical')


path=f'{data}/test'
image_datagentest=data_gen.flow_from_directory(path,target_size=(128,130),class_mode='categorical')




Optimizer_str_to_func = {
    'Adam' : tf.keras.optimizers.Adam,
    'Nadam' : tf.keras.optimizers.Nadam,
    'RMSprop' : tf.keras.optimizers.RMSprop,
    'SGD' : tf.keras.optimizers.SGD
    }


initial_L2 = 0.0005
initial_lr = 0.0005
epochs = 25

def GenreModel(hp_space, input_shape = (128,130,3), classes=10):
    K.set_learning_phase(1)
    K.set_image_data_format('channels_last')
        
    # tensorboard_callback = TensorBoard(log_dir=f'logs/{model_name}')
    
    model_id = str(uuid.uuid4())[:5]  # Unique 5-charachter model ID, different every time the function is called
    
    
    ks = int(hp_space['kern_size'])
    
    if hp_space['pooling'] == 'max':
        pooling_type = MaxPooling2D(pool_size=(2,2), strides=(2,2))
    elif hp_space['pooling'] == 'avg':
        pooling_type = AveragePooling2D(pool_size=(2,2), strides=(2,2))


    X_input = Input(input_shape)
    
    X = Conv2D(filters=8, kernel_size=(ks,ks), padding='same',
               input_shape=(128, 130, 3),
               kernel_regularizer=l2(initial_L2 * hp_space['l2_mult']))(X_input)
    if hp_space['use_BN']:
        X = BatchNormalization(axis=3)(X)
    X = Activation(hp_space['activation'])(X)
    X = pooling_type(X)
    
        

    X = Conv2D(filters=16, kernel_size=(ks,ks), padding='same',
               kernel_regularizer=l2(initial_L2 * hp_space['l2_mult']))(X)
    if hp_space['use_BN']:
        X = BatchNormalization(axis=3)(X)
    X = Activation(hp_space['activation'])(X)
    X = pooling_type(X)
    


    X = Conv2D(filters=64, kernel_size=(ks,ks), padding='same',
               kernel_regularizer=l2(initial_L2 * hp_space['l2_mult']))(X)
    if hp_space['use_BN']:
        X = BatchNormalization(axis=3)(X)
    X = Activation(hp_space['activation'])(X)
    X = pooling_type(X)
    
    if hp_space['dropout_conv'] is not None:
        X = Dropout(hp_space['dropout_conv'])(X)

    X = Conv2D(filters=128, kernel_size=(ks,ks), padding='same',
               kernel_regularizer=l2(initial_L2 * hp_space['l2_mult']))(X)
    if hp_space['use_BN']:
        X = BatchNormalization(axis=-1)(X)
    X = Activation(hp_space['activation'])(X)
    X = pooling_type(X)
    
    if hp_space['dropout_conv'] is not None:
        X = Dropout(hp_space['dropout_conv'])(X)
    
    X = Conv2D(filters=128, kernel_size=(ks,ks), padding='same',
               kernel_regularizer=l2(initial_L2 * hp_space['l2_mult']))(X)
    if hp_space['use_BN']:
        X = BatchNormalization(axis=-1)(X)
    X = Activation(hp_space['activation'])(X)
    X = pooling_type(X)
    
    if hp_space['dropout_conv'] is not None:
        X = Dropout(hp_space['dropout_conv'])(X)    

    X = Flatten()(X)

    X = Dropout(hp_space['dropout'])(X)

    X = (Dense(512, activation=hp_space['activation']))(X)
    X = (Dense(256, activation=hp_space['activation']))(X)
    X = (Dense(10, activation='softmax'))(X)

    model = Model(X_input, X)
        
    model.compile(optimizer=Optimizer_str_to_func[hp_space['optimizer']](learning_rate=initial_lr * hp_space['lr_mult']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(image_datagentrain,
                        epochs=epochs,
                        batch_size=128,
                        validation_data=image_datagentest
                        # ,callbacks=[tensorboard_callback]
                        ).history
    
    K.set_learning_phase(0)
    
    max_val_acc = np.max(history['val_accuracy'])
    
    model_name = "model_{}_{}".format(str(max_val_acc), model_id)
    
    results = {
        # fmin function in hyperopt minimized 'loss' by default
        
        # metrics
        'loss' : -max_val_acc,
        'best_accuracy' : max(history['accuracy']),
        'best_actual_loss' : min(history['loss']),
        'best_val_accuracy' : max(history['val_accuracy']),
        'best_val_loss' : min(history['val_loss']),
        # Misc
        'status' : STATUS_OK,
        'history' : history,
        'space' : hp_space
        }
    
    return model, model_name, results






