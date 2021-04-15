# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Want to create a model function that take a hyperparameter space and builds a model
based on that space.

Create a fucntion that trains the model on the dataset and return the validation
loss either as a single scalar or as value in a dictionary with corresponding key 'loss'

This function will then be used in the optimize function which will output negative of the max validation loss
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization,
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D,
                          GlobalMaxPooling2D, Dropout)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K
import os
import random
from hyperopt import STATUS_OK, STATUS_FAIL
import shutil
import uuid




data = '../GTZAN_Dataset'
genres = list(os.listdir(f'{data}/images_original/'))
genres.remove('.DS_Store')
print(genres)

directory = f"{data}/images_original/"
for g in genres:
    filenames = os.listdir(os.path.join(directory,f"{g}"))
    random.shuffle(filenames)
    test_files = filenames[0:30]
    # Only need to run the following once to split the dataset
    #os.makedirs(f'{data}/images_test/' + f'{g}')  
    #for f in test_files:
        #shutil.move(directory + f"{g}"+ "/" + f,f"{data}/images_test/" + f"{g}")


data_gen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
path=f'{data}/images_original' 
image_datagentrain=data_gen.flow_from_directory(path,target_size=(288,432),color_mode="rgba",batch_size=1,class_mode='categorical')


path=f'{data}/images_test'
image_datagentest=data_gen.flow_from_directory(path,target_size=(288,432),color_mode="rgba",batch_size=1,class_mode='categorical')


INIT_L2 = 0.0007

Optimizer_str_to_func = {
    'Adam' : tf.keras.optimizers.Adam,
    'Nadam' : tf.keras.optimizers.Nadam,
    'RMSprop' : tf.keras.optimizers.RMSprop,
    'SGD' : tf.keras.optimizers.SGD
    }

def get_f1(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def compile_fit_GenreModel(hp_space, input_shape = (288,432,4), classes=10):
    K.set_learning_phase(1)
    K.set_image_data_format('channels_last')
    
    dropout_rate = hp_space['dr']
    lr_mult = hp_space['lr_mult']
    opt = hp_space['optimizer']
    
    model_id = str(uuid.uuid4())[:5]  # Unique 5-charachter model ID, different every time the function is called
    
    np.random.seed(10)
    X_input = Input(input_shape)
      
    X = Conv2D(8,kernel_size=(3,3),strides=(1,1),
               kernel_initializer = glorot_uniform(seed=9),
               kernel_regularizer=tf.keras.regularizers.l2(INIT_L2 * hp_space['l2_mult']))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation(hp_space['activation'])(X)
    X = MaxPooling2D((2,2))(X)
    
    X = Conv2D(16,kernel_size=(3,3),strides = (1,1),
               kernel_initializer=glorot_uniform(seed=9),
               kernel_regularizer=tf.keras.regularizers.l2(INIT_L2 * hp_space['l2_mult']))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(hp_space['activation'])(X)
    X = MaxPooling2D((2,2))(X)
    
    X = Conv2D(32,kernel_size=(3,3),strides = (1,1),
               kernel_initializer = glorot_uniform(seed=9),
               kernel_regularizer=tf.keras.regularizers.l2(INIT_L2 * hp_space['l2_mult']))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(hp_space['activation'])(X)
    X = MaxPooling2D((2,2))(X)
      
    X = Conv2D(64,kernel_size=(3,3),strides=(1,1),
               kernel_initializer=glorot_uniform(seed=9),
               kernel_regularizer=tf.keras.regularizers.l2(INIT_L2 * hp_space['l2_mult']))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation(hp_space['activation'])(X)
    X = MaxPooling2D((2,2))(X)
      
    X = Dropout(dropout_rate)(X)
    
    X = Flatten()(X)
      
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=9))(X)
      
    model = Model(inputs=X_input,outputs=X,name='GenreModel')
    
    
    model.compile(optimizer=Optimizer_str_to_func[opt](learning_rate=0.001 * lr_mult),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', get_f1])
    
    history = model.fit_generator(image_datagentrain,
                                  epochs=10,
                                  validation_data=image_datagentest).history
    
    K.set_learning_phase(0)
    
    max_val_acc = np.max(history['val_accuracy'])
    
    model_name = "mdoel_{}_{}".format(str(max_val_acc), model_id)
    
    results = {
        # fmin function in hyperopt minimized 'loss' by default
        # metrics
        'loss' : -max_val_acc,
        'accuracy' : history['accuracy'],
        'actual_loss' : history['loss'],
        'val_ accuracy' : history['val_accuracy'],
        'val_loss' : history['val_loss'],
        # Misc
        'status' : STATUS_OK,
        'space' : hp_space
        }
    
    return model, model_name, results


def base_GenreModel():
    
    # Starting hyperparameter values that will later be optimized
    base_space = {
    'lr_mult' : 1.0,
    'dr' : 0.3,
    'optimizer' : 'Adam',
    'l2_mult' : 1.0,
    'activation' : 'relu'
    }
    
    compile_fit_GenreModel(base_space)
    
    return















