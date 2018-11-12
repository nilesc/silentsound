from wavegan.wavegan import WaveGANGenerator
import tensorflow as tf
import numpy as np
import PIL.Image
from IPython.display import display, Audio
import time as time
import librosa

def build_generator(graph):
    with graph.as_default():
#        placeholder = tf.placeholder(dtype=float, shape=(None, 10), name='video_input')
#        next_layer = tf.layers.dense(inputs=placeholder, units=100)

        rest_of_network = WaveGANGenerator(next_layer)

if __name__ == '__main__':

    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('infer.meta')
    sess = tf.InteractiveSession()
    graph = tf.get_default_graph()
    build_generator(graph)
    saver.restore(sess, 'model.ckpt')

    # CHANGE THESE to change number of examples generated/displayed
    ngenerate = 64
    ndisplay = 64

    # Sample latent vectors
    _z = (np.random.rand(ngenerate, 10) * 2.) - 1.

    # Generate
    video_input = graph.get_tensor_by_name('video_input:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
    G_z_spec = graph.get_tensor_by_name('G_z_spec:0')

    _G_z = sess.run([video_input, G_z], {video_input: _z})

    #for i in range(ndisplay):
    #    librosa.output.write_wav('sample_audio/{}.wav'.format(i), _G_z[i], 16000) 
