from wavegan.wavegan import WaveGANGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Reshape, ReLU, Conv2DTranspose
import numpy as np
import PIL.Image
from IPython.display import display, Audio
import time as time

#class RestoreCkptCallback(keras.callbacks.Callback):
#    def __init__(self, pretrained_file):
#        self.pretrained_file = pretrained_file
#        self.sess = keras.backend.get_session()
#        self.saver = tf.train.Saver()
#    def on_train_begin(self, logs=None):
#        if self.pretrian_model_path:
#            self.saver.restore(self.sess, self.pretrian_model_path)

def build_generator():

    model = Sequential()
    # FC and reshape for convolution
    model.add(Dense(1024, input_shape=(100,)))
    model.add(Reshape((16, 64)))
    model.add(ReLU())

    model.add(Conv2DTransponse(
        tf.expand_dims(inputs, axis_1)),
        8 * 64,


if __name__ == '__main__':
    build_generator()
