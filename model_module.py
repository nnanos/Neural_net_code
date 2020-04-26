import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras import regularizers

def create_model(z):

    if z==1:
        str = 'sigmoid'
    else:
        str = 'relu'

    model = tf.keras.models.Sequential()
    #defining the first input layer (visible layer) and the first hidden layer with the above line of code
    model.add(tf.keras.layers.Dense(10, activation=str, activity_regularizer= regularizers.l1(0.1),input_dim = 943))
    #model.add(tf.keras.layers.Dense(10, activation = str))
    #model.add(tf.keras.layers.Dense(10, activation=str))
    model.add(tf.keras.layers.Dense( 1682 , activation = str ))

    history = model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.6, nesterov=False),
                  loss='mse',
                  metrics=['mean_squared_error', 'mean_absolute_error'])

    return model