from wavegan.wavegan import WaveGANGenerator
import tensorflow as tf
import numpy as np
import PIL.Image
from IPython.display import display, Audio
import time as time
import librosa

def build_generator(graph):

        rest_of_network = WaveGANGenerator(next_layer)

if __name__ == '__main__':

    tf.reset_default_graph()

    sess = tf.InteractiveSession()
    graph = tf.get_default_graph()
    video_input = tf.placeholder(dtype=float, shape=(None, 1), name='video_input')
    next_layer = tf.layers.dense(inputs=video_input, units=100)

    saver = tf.train.import_meta_graph('infer.meta', input_map={'z': next_layer})
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, 'model.ckpt')

    # CHANGE THESE to change number of examples generated/displayed
    ngenerate = 64
    ndisplay = 64

    # Sample latent vectors
    _z = (np.random.rand(ngenerate, 10) * 2.) - 1.

    # Generate
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
    G_z_spec = graph.get_tensor_by_name('G_z_spec:0')

    _G_z = sess.run(G_z, {video_input: _z})

    for i in range(ndisplay):
        librosa.output.write_wav('sample_audio/{}.wav'.format(i), _G_z[i], 16000) 
