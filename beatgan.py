"""
Based on code from BeatGan: https://github.com/NarainKrishnamurthy/BeatGAN2.0
"""
from __future__ import print_function
from __future__ import division

from keras import backend as K
K.set_image_dim_ordering('th') # ensure our dimension notation matches

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Reshape
from keras.models import Model
from keras.layers.merge import Concatenate, _Merge
from keras.layers.core import Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv1D, Conv3D
from keras.layers.convolutional import Convolution2D, AveragePooling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras import utils
import numpy as np
from scipy.io import wavfile
from PIL import Image, ImageOps
from functools import partial
import random
import argparse
import math
import wavfile24
import os
import os.path
import glob

NP_RANDOM_SEED = 2000

# Set Model Hyperparameters
class BeatGanHyperParameters():
    def __init__(self, num_channels, batch_size, model_size, D_update_per_G_update):
        self.c = num_channels
        self.b = batch_size
        self.d = model_size
        self.D_updates_per_G_update = D_update_per_G_update
        self.WGAN_GP_weight = 10


def get_generator(num_dimensions, num_channels):
    model_input = Input(shape=(5, 5, 5, num_channels))
    # Change input_dim to be the size of our video
    model = Conv3D(16, 5, strides=1, padding='valid', data_format='channels_last')(model_input)
    model = Flatten()(model)
    model = Dense(units=256*num_dimensions)(model)
    # Add layers here to connect video_size to the 100 units
    model = Reshape((1, 16, 16*num_dimensions), input_shape = (256*num_dimensions,))(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(8*num_dimensions, (1,25), strides=(1,4), padding="same", data_format='channels_last')(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(4*num_dimensions, (1,25), strides=(1,4), padding="same", data_format='channels_last')(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(2*num_dimensions, (1,25), strides=(1,4), padding="same", data_format='channels_last')(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(num_dimensions, (1,25), strides=(1,4), padding="same", data_format='channels_last')(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(num_channels, (1,25), strides=(1,4), padding="same", data_format='channels_last')(model)
    model = Activation('tanh')(model)
    model = Reshape((16384, num_channels), input_shape = (1, 16384, num_channels))(model)

    return Model(inputs=model_input, outputs=(model, model_input))

def get_discriminator(num_dimensions, num_channels):
    audio_model_input = Input(shape=(16384, num_channels))

    audio_model = Conv1D(num_dimensions, 25, strides=4, padding="same", input_shape=(16384, num_channels))(audio_model_input)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(2*num_dimensions, 25, strides=4, padding="same")(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(4*num_dimensions, 25, strides=4, padding="same")(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(8*num_dimensions, 25, strides=4, padding="same")(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(16*num_dimensions, 25, strides=4, padding="same")(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Reshape((256*num_dimensions, ), input_shape = (1, 16, 16*num_dimensions))(audio_model)

    video_model_input = Input(shape=(5, 5, 5, num_channels))
    video_model = Conv3D(16, 5, strides=1, padding='valid', data_format='channels_last')(video_model_input)
    video_model = Flatten()(video_model)

    final_model = Concatenate()([audio_model, video_model])
    final_model = Dense(256)(final_model)
    final_model = Dense(1)(final_model)

    return Model(inputs=[audio_model_input, video_model_input], outputs=final_model)

def generator_containing_discriminator(generator, discriminator):
    model = Input(shape=(5, 5, 5, 2))
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
        weights = K.random_uniform((64, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def generate_after_training(BATCH_SIZE):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('goodgenerator.h5')

    noise = np.zeros((BATCH_SIZE, 100))
    for i in range(BATCH_SIZE):
        noise[i, :] = np.random.uniform(-1, 1, 100)
    generated_audio = generator.predict(noise, verbose=1)
    print(generated_audio.shape)
    for audio in generated_audio:
        wavfile.write('thing.wav', 14700, audio)

def make_generator_model(generator, discriminator, num_channels):
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False

    generator_input = Input(shape=(5, 5, 5, num_channels))
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])

    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)
    return generator_model

def make_discriminator_model(generator, discriminator, num_channels):
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    real_samples = Input(shape=(16384, num_channels))
    generator_input_for_discriminator = Input(shape=(5, 5, 5, num_channels))
    print(generator_input_for_discriminator)
    print(generator_input_for_discriminator.shape)
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    # Need to modify real_samples to include original input
    discriminator_output_from_real_samples = discriminator([real_samples, generator_input_for_discriminator])
    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator[0]])
    print([real_samples, generated_samples_for_discriminator[0]])
    print(averaged_samples)
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

def load_videos(x):
    return np.ones((10, 16384, 10)), np.ones((10, 5, 5, 5, 10))

def train(epochs, BATCH_SIZE):
    np.random.seed(NP_RANDOM_SEED)
    X_train_audio, X_train_video = load_videos(1)
    # np.random.shuffle(X_train_audio)

    num_dimensions = 10
    num_channels = 10
    discriminator = get_discriminator(num_dimensions, num_channels)
    generator = get_generator(num_dimensions, num_channels)

    generator_model = make_generator_model(generator, discriminator, num_channels)
    discriminator_model = make_discriminator_model(generator, discriminator, num_channels)


    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    print("Number of batches", int(X_train_audio.shape[0]/BATCH_SIZE))
    for epoch in range(epochs):
        print("Epoch is", epoch)
        dl, gl = {}, {}
        # np.random.shuffle(X_train)
        for index in range(int(X_train_audio.shape[0]/BATCH_SIZE)):
            audio_batch = X_train_audio[index*BATCH_SIZE:(index+1)*BATCH_SIZE].reshape(BATCH_SIZE, 16384, num_channels)
            video_batch = X_train_video[index*BATCH_SIZE:(index+1)*BATCH_SIZE].reshape(BATCH_SIZE, 5, 5, 5, 10)
            # noise = get_noise((BATCH_SIZE, 100))
            d_loss = discriminator_model.train_on_batch([audio_batch, video_batch], [positive_y, negative_y, dummy_y])
            dl = d_loss
            if index % hp.D_updates_per_G_update == 0:
                #print("batch %d d_loss : %s" % (index, d_loss))
                noise = get_noise((BATCH_SIZE, 100))
                g_loss = generator_model.train_on_batch(video_batch, positive_y)
                gl = g_loss
                #print("batch %d g_loss : %0.10f" % (index, g_loss))

        if epoch % 500 == 0:
            print("epoch %d d_loss : %s" % (epoch, dl))
            print("epoch %d g_loss : %0.10f" % (epoch, gl))
            generator.save_weights('weights/generator' + str(epoch) + '.h5', True)
            discriminator.save_weights('weights/discriminator' + str(epoch) + '.h5', True)
            generate_one(generator, epoch, 0)

def generate_one(generator, epoch, index):
    noise = get_noise((1,100))
    generated_audio = generator.predict(noise, verbose=1)
    q = np.array(generated_audio[0]*8388608).astype('int32')
    wavfile24.write('outputs/epoch' + ("%04d" % epoch) + 'index'+ ("%03d" % index) + '.wav', 14700, q, bitrate=24)

def generate_batch(generator, weights_file, batch_size):
    noise = get_noise((batch_size,100))
    generator.load_weights(weights_file)
    generated_audio = generator.predict(noise, verbose=1)
    re_normalization_factor = 8388608
    assumed_sample_length = 14112
    sample_rate = 14700
    for i in range(len(generated_audio)):
        output = generated_audio[i]
        q = np.array(output*re_normalization_factor).astype('int32')
        wavfile24.write('generated_outputs/output' + ("%03d" % i) + '.wav', sample_rate, np.concatenate((q[:assumed_sample_length], q[:assumed_sample_length])), bitrate=24)

# train(6100, hp.b) - this was the original training call, 6k epochs


# generator = get_generator()
# generate_batch(generator, 'weights/generator6000.h5', 40)
# print (compute_similarity_score(0.10))
#
# # Test Script that lets you manually check similarity of a generated output vs the training set
# original_beats = load_beat_data(0)
# X_train = load_beat_data(1)
# generated_outputs = glob.glob(os.path.normpath('/home/narainsk/beat_gan/BeatsByGAN/generated_outputs/*.wav'))
# generated_output_file = generated_outputs[15]
# print ('using file' + generated_output_file)
# normalization_factor = 8388608
# num_samples_compared = 14112
# b = (wavfile24.read(generated_output_file)[1])/normalization_factor
# for i in range(len(original_beats)):
#     a = X_train[i*5]
#     error = np.sum(np.square(a[:num_samples_compared] - b[:num_samples_compared]))
#     similarity = error/(np.sum(np.square(a[:num_samples_compared])))
#     if similarity < 0.5:
#         print (i)
#         print (similarity)
#         print (original_beats[i][1][:6])
#         wavfile24.write('similarities_test/similar' + str(i) + '.wav', 44100, original_beats[i][1] , bitrate=24)
#         wavfile24.write('similarities_test/similar' + str(i) + 'downsampled.wav', 14700, original_beats[i][1][::3] , bitrate=24)


def load_wavegan_paper_drumhit_data(policy):
    print("Loading data")
    X_train = []
    skip_list = set(['/home/narainsk/beat_gan/BeatsByGAN/drums/Roland JV 1080/MaxV - Guiro.wav'])
    normalization_factor = 32768
    paths = glob.glob(os.path.normpath(os.getcwd() + '/drums/*/*.wav'))
    for i in range(len(paths)):
        if paths[i] not in skip_list:
            sound = wavfile.read(paths[i])
            if policy == 0:
                X_train.append(sound)
            elif policy == 1:
                if sound[1].size <= 44100:
                    wavfile.write('temp.wav', 14700, sound[1][::3])
                    temp = wavfile.read('temp.wav')
                    normed = np.concatenate((temp[1], np.zeros(16384 - len(temp[1]))))/normalization_factor
                    X_train.append(normed)
    return np.array(X_train) if policy == 1 else X_train
# X_train = load_wavegan_paper_drumhit_data(1)
# np.random.shuffle(X_train)
#
# wavfile24.write('a.wav', 44100, X_train[0][1], bitrate=24)

