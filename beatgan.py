"""
Based on code from BeatGan: https://github.com/NarainKrishnamurthy/BeatGAN2.0
"""
from __future__ import print_function
from __future__ import division

from keras import backend as K
K.set_image_dim_ordering('th') # ensure our dimension notation matches

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Reshape
from keras.layers.merge import Concatenate, _Merge
from keras.layers.core import Activation, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv1D, Conv3D
from keras.layers.convolutional import Convolution2D, AveragePooling2D, Conv2DTranspose, MaxPooling3D
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
from keras import utils
import numpy as np
from functools import partial
import random
import os
import os.path
import glob
import pickle
import csv

NP_RANDOM_SEED = 2000
train_data = 'train_inputs'
test_data = 'test_inputs'
audio_length = 16384
regularization_penalty = 0.01

# Set Model Hyperparameters
class HyperParameters():
    def __init__(self, num_channels, batch_size, model_size, D_update_per_G_update, window_radius, num_frames):
        self.c = num_channels
        self.b = batch_size
        self.d = model_size
        self.D_updates_per_G_update = D_update_per_G_update
        self.WGAN_GP_weight = 10
        self.window_radius = window_radius
        self.num_frames = num_frames
        self.video_shape = (num_frames, 2*window_radius, 2*window_radius, 3) # (5, 50, 50, 3)

hp = HyperParameters(1, 5, 5, 100, 25, 10)


def get_generator(wavegan_instance):
    model_input = Input(shape=hp.video_shape)
    input_copied = Lambda(lambda x: x, input_shape=model_input.shape[1:])(model_input)

    # Change below here
    model = Conv3D(filters=16, kernel_size=3, strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model_input)
    model = Activation('relu')(model)
    model = Conv3D(filters=16, kernel_size=3, strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model)
    model = MaxPooling3D(pool_size=(2, 2, 2), strides=1, padding='valid', data_format='channels_last')(model)
    model = Conv3D(filters=32, kernel_size=3, strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model)
    model = Activation('relu')(model)
    model = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model)
    model = Activation('relu')(model)
    model = Conv3D(filters=64, kernel_size=(1, 3, 3), strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model)
    model = Activation('relu')(model)
    model = Conv3D(filters=64, kernel_size=(1, 3, 3), strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model)
    model = MaxPooling3D(pool_size=(2, 2, 2), strides=1, padding='valid', data_format='channels_last')(model)
    model = Flatten()(model)
    model = Dense(1024, activation='relu')(model)

    # Change above here
    model = wavegan_instance(model)

    return Model(inputs=model_input, outputs=(model, input_copied))

def get_wavegan():
    model_input = Input(shape=(1024,))
    model = Dense(units=256*hp.d)(model_input)
    # Add layers here to connect video_size to the 100 units
    model = Reshape((1, 16, 16*hp.d), input_shape = (256*hp.d,))(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(8*hp.d, (1,25), strides=(1,4), padding="same", data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(4*hp.d, (1,25), strides=(1,4), padding="same", data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(2*hp.d, (1,25), strides=(1,4), padding="same", data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(hp.d, (1,25), strides=(1,4), padding="same", data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(hp.c, (1,25), strides=(1,4), padding="same", data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(model)
    model = Activation('tanh')(model)
    model = Reshape((16384, hp.c), input_shape = (1, 16384, hp.c))(model)

    return Model(inputs=model_input, outputs=model)

def get_discriminator():
    audio_model_input = Input(shape=(16384, hp.c))
    audio_model = Conv1D(hp.d, 25, strides=4, padding="same", input_shape=(16384, hp.c), kernel_regularizer=l2(regularization_penalty))(audio_model_input)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(2*hp.d, 25, strides=4, padding="same", kernel_regularizer=l2(regularization_penalty))(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(4*hp.d, 25, strides=4, padding="same", kernel_regularizer=l2(regularization_penalty))(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(8*hp.d, 25, strides=4, padding="same", kernel_regularizer=l2(regularization_penalty))(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(16*hp.d, 25, strides=4, padding="same", kernel_regularizer=l2(regularization_penalty))(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Reshape((256*hp.d, ), input_shape = (1, 16, 16*hp.d))(audio_model)

    video_model_input = Input(shape=(hp.video_shape))
    # Change below here
    video_model = Conv3D(filters=16, kernel_size=3, strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(video_model_input)
    video_model = Activation('relu')(video_model)
    video_model = Conv3D(filters=16, kernel_size=3, strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(video_model)
    video_model = MaxPooling3D(pool_size=(2, 2, 2), strides=1, padding='valid', data_format='channels_last')(video_model)
    video_model = Conv3D(filters=32, kernel_size=3, strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(video_model)
    video_model = Activation('relu')(video_model)
    video_model = Conv3D(filters=32, kernel_size=(1, 3, 3), strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(video_model)
    video_model = Activation('relu')(video_model)
    video_model = Conv3D(filters=64, kernel_size=(1, 3, 3), strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(video_model)
    video_model = Activation('relu')(video_model)
    video_model = Conv3D(filters=64, kernel_size=(1, 3, 3), strides=1, padding='valid', data_format='channels_last', kernel_regularizer=l2(regularization_penalty))(video_model)
    video_model = MaxPooling3D(pool_size=(2, 2, 2), strides=1, padding='valid', data_format='channels_last')(video_model)
    video_model = Flatten()(video_model)
    video_model = Dense(1024, activation='relu')(video_model)
    # Change above here

    final_model = Concatenate()([audio_model, video_model])
    # Change below here
    final_model = Dense(256, kernel_regularizer=l2(regularization_penalty))(final_model)
    final_model = Dense(256, kernel_regularizer=l2(regularization_penalty))(final_model)
    final_model = Dense(128, kernel_regularizer=l2(regularization_penalty))(final_model)
    final_model = Dense(64, kernel_regularizer=l2(regularization_penalty))(final_model)
    # Change above here
    final_model = Dense(1)(final_model)

    return Model(inputs=[audio_model_input, video_model_input], outputs=final_model)

def generator_containing_discriminator(generator, discriminator):
    model = Input(shape=(hp.video_shape))
    model = generator(model)
    model = discriminator(model)
    return model

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        shape = inputs[0].get_shape().as_list()
        weights = K.random_uniform((hp.b, shape[1], shape[2]))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def make_generator_model(generator, discriminator):
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False

    generator_input = Input(shape=hp.video_shape)
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])

    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)
    return generator_model

def make_discriminator_model(generator, discriminator):
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    real_samples = Input(shape=(16384, hp.c))
    generator_input_for_discriminator = Input(shape=hp.video_shape)
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    # Need to modify real_samples to include original input
    discriminator_output_from_real_samples = discriminator([real_samples, generator_input_for_discriminator])
    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator[0]])
    averaged_samples_out = discriminator([averaged_samples, generator_input_for_discriminator])

    partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=10)
    partial_gp_loss.__name__ = 'gradient_penalty'

    discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])

    discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])
    return discriminator_model

def get_noise(shape):
    return np.random.uniform(-1, 1, shape).astype(np.float32)

def pad_or_truncate(array, length):
    if len(array) > length:
        return array[:length]

    return_val = np.zeros((length,))
    return_val[:len(array)] = array

    return return_val

def crop_videos(video, num_frames, x, y, crop_window_radius):
    center_x = int(video.shape[1] * float(x))
    center_y = int(video.shape[2] * float(y))

    x_length = video.shape[1] - 1
    y_length = video.shape[2] - 1

    x_low = min(center_x - crop_window_radius, 0)
    x_high = max(center_x + crop_window_radius, x_length)
    y_low = min(center_y - crop_window_radius, 0)
    y_high = max(center_y + crop_window_radius, y_length)

    video = np.pad(video, ((0, 0), (-x_low, x_high - x_length), (-y_low, y_high - y_length), (0, 0)), mode='edge')

    center_x -= x_low
    center_y -= y_low
    frames_per_sec = 30
    first_sec = video[:frames_per_sec]
    frame_indices = np.linspace(0, first_sec.shape[0] - 1, num_frames).astype(int)

    reduced_frames = first_sec[frame_indices]

    return reduced_frames[:,
            center_x-crop_window_radius:center_x+crop_window_radius,
            center_y-crop_window_radius:center_y+crop_window_radius,
            :]

def load_videos(filename, window_radius):
    audio = []
    videos = []
    video_ids = []
    for filename in os.listdir(train_data):
        with open('{}/{}'.format(train_data, filename), 'rb') as opened_file:
            unpickler = pickle.Unpickler(opened_file)
            row = None
            while True:
                try:
                    row = unpickler.load()
                except EOFError:
                    break

                audio.append(pad_or_truncate(row[0], audio_length))
                videos.append(crop_videos(row[1], hp.num_frames, row[2], row[3], window_radius))
                video_ids.append(row[5])

    return np.asarray(audio), np.asarray(videos), video_ids

def train(epochs):
    np.random.seed(NP_RANDOM_SEED)
    X_train_audio, X_train_video, _ = load_videos(train_data, hp.window_radius)
    shuffled_indices = np.arange(len(X_train_audio))
    np.random.shuffle(shuffled_indices)
    X_train_audio = X_train_audio[shuffled_indices]
    X_train_video = X_train_video[shuffled_indices]

    discriminator = get_discriminator()
    wavegan_instance = get_wavegan()
    generator = get_generator(wavegan_instance)

    generator_model = make_generator_model(generator, discriminator)
    generator.summary()
    discriminator_model = make_discriminator_model(generator, discriminator)
    discriminator.summary()


    positive_y = np.ones((hp.b, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((hp.b, 1), dtype=np.float32)

    print("Number of batches", int(X_train_audio.shape[0]/hp.b))
    for epoch in range(epochs):
        print("Epoch is", epoch)
        dl, gl = {}, {}
        shuffled_indices = np.arange(len(X_train_audio))
        np.random.shuffle(shuffled_indices)
        X_train_audio = X_train_audio[shuffled_indices]
        X_train_video = X_train_video[shuffled_indices]

        for index in range(int(X_train_audio.shape[0]/hp.b)):
            print(X_train_audio[index*hp.b:(index+1)*hp.b].shape)
            audio_batch = X_train_audio[index*hp.b:(index+1)*hp.b].reshape(hp.b, 16384, hp.c)
            video_batch = X_train_video[index*hp.b:(index+1)*hp.b].reshape((hp.b,) + hp.video_shape)
            d_loss = discriminator_model.train_on_batch([audio_batch, video_batch], [positive_y, negative_y, dummy_y])
            dl = d_loss
            if index % hp.D_updates_per_G_update == 0:
                noise = get_noise((hp.b, 100))
                g_loss = generator_model.train_on_batch(video_batch, positive_y)
                gl = g_loss

        if epoch % 200 == 0:
            print("epoch %d d_loss : %s" % (epoch, dl))
            print("epoch %d g_loss : %0.10f" % (epoch, gl))

            with open('weights/losses.csv', mode='a+') as loss_file:
                loss_writer = csv.writer(loss_file)
                loss_writer.writerow([epoch, dl, gl])

            generator.save_weights('weights/generator' + str(epoch) + '.h5', True)
            discriminator.save_weights('weights/discriminator' + str(epoch) + '.h5', True)
