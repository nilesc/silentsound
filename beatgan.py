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
import pickle

NP_RANDOM_SEED = 2000
train_data = 'train_condensed.pkl'
test_data = 'test_condensed.pkl'
audio_length = 16384

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
        self.video_shape = (num_frames, 2*window_radius, 2*window_radius, 3)

hp = HyperParameters(1, 20, 5, 100, 50, 10)


def get_generator():
    model_input = Input(shape=hp.video_shape)
    # Change input_dim to be the size of our video
    model = Conv3D(16, 5, strides=1, padding='valid', data_format='channels_last')(model_input)
    model = Flatten()(model)
    model = Dense(units=256*hp.d)(model)
    # Add layers here to connect video_size to the 100 units
    model = Reshape((1, 16, 16*hp.d), input_shape = (256*hp.d,))(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(8*hp.d, (1,25), strides=(1,4), padding="same", data_format='channels_last')(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(4*hp.d, (1,25), strides=(1,4), padding="same", data_format='channels_last')(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(2*hp.d, (1,25), strides=(1,4), padding="same", data_format='channels_last')(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(hp.d, (1,25), strides=(1,4), padding="same", data_format='channels_last')(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(hp.c, (1,25), strides=(1,4), padding="same", data_format='channels_last')(model)
    model = Activation('tanh')(model)
    model = Reshape((16384, hp.c), input_shape = (1, 16384, hp.c))(model)

    return Model(inputs=model_input, outputs=(model, model_input))

def get_discriminator():
    audio_model_input = Input(shape=(16384, hp.c))

    audio_model = Conv1D(hp.d, 25, strides=4, padding="same", input_shape=(16384, hp.c))(audio_model_input)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(2*hp.d, 25, strides=4, padding="same")(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(4*hp.d, 25, strides=4, padding="same")(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(8*hp.d, 25, strides=4, padding="same")(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Conv1D(16*hp.d, 25, strides=4, padding="same")(audio_model)
    audio_model = LeakyReLU(alpha=0.2)(audio_model)
    audio_model = Reshape((256*hp.d, ), input_shape = (1, 16, 16*hp.d))(audio_model)

    video_model_input = Input(shape=(hp.video_shape))
    video_model = Conv3D(16, 5, strides=1, padding='valid', data_format='channels_last')(video_model_input)
    video_model = Flatten()(video_model)

    final_model = Concatenate()([audio_model, video_model])
    final_model = Dense(256)(final_model)
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

def generate_after_training():
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('goodgenerator.h5')

    noise = np.zeros((hp.b, 100))
    for i in range(hp.b):
        noise[i, :] = np.random.uniform(-1, 1, 100)
    generated_audio = generator.predict(noise, verbose=1)
    print(generated_audio.shape)
    for audio in generated_audio:
        wavfile.write('thing.wav', 14700, audio)

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
    frame_indices = np.linspace(0, frames_per_sec - 1, num_frames).astype(int)

    cropped = video[:frames_per_sec,
            center_x-crop_window_radius:center_x+crop_window_radius,
            center_y-crop_window_radius:center_y+crop_window_radius,
            :]

    return cropped[frame_indices]

def load_videos(filename, window_radius):
    audio = []
    videos = []
    with open(filename, 'rb') as opened_file:
        all_info = pickle.load(opened_file)
        for row in all_info:
            audio.append(pad_or_truncate(row[0], audio_length))
            videos.append(crop_videos(row[1], hp.num_frames, row[2], row[3], window_radius))

    for video in videos:
        print(video.shape)

    return np.asarray(audio), np.asarray(videos)
    # return np.ones((1000, 16384, 10)), np.ones((1000, 5, 5, 5, 10))

def train(epochs):
    np.random.seed(NP_RANDOM_SEED)
    X_train_audio, X_train_video = load_videos(test_data, hp.window_radius)
    # np.random.shuffle(X_train_audio)

    discriminator = get_discriminator()
    generator = get_generator()

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
        # np.random.shuffle(X_train)
        for index in range(int(X_train_audio.shape[0]/hp.b)):
            audio_batch = X_train_audio[index*hp.b:(index+1)*hp.b].reshape(hp.b, 16384, hp.c)
            video_batch = X_train_video[index*hp.b:(index+1)*hp.b].reshape((hp.b,) + hp.video_shape)
            # noise = get_noise((BATCH_SIZE, 100))
            d_loss = discriminator_model.train_on_batch([audio_batch, video_batch], [positive_y, negative_y, dummy_y])
            dl = d_loss
            if index % hp.D_updates_per_G_update == 0:
                #print("batch %d d_loss : %s" % (index, d_loss))
                noise = get_noise((hp.b, 100))
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

def generate_batch(generator, weights_file):
    noise = get_noise((hp.b,100))
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

