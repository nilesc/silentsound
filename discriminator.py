from wavegan.wavegan import WaveGANGenerator
import tensorflow as tf
import numpy as np
import PIL.Image
from IPython.display import display, Audio
import time as time
import librosa

def build_discriminator(graph, input_audio, input_video):
    with graph.as_default():
        next_layer = tf. 


if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    graph = tf.get_default_graph()
    inputs = np.rand.generate(2, 16384)

    build_discriminator(graph, inputs[0], inputs[1])
