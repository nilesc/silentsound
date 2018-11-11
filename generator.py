from wavegan.wavegan import WaveGANGenerator
import tensorflow as tf
import numpy as np
import PIL.Image
from IPython.display import display, Audio
import time as time
import librosa

def build_generator(graph):
    with graph.as_default():
        placeholder = tf.placeholder(dtype=float, shape=(1, 100), name='video_input')
        next_layer = tf.layers.dense(inputs=placeholder, units=10)

        rest_of_network = WaveGANGenerator(next_layer)

    return rest_of_network

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
    _z = (np.random.rand(ngenerate, 100) * 2.) - 1.

    # Generate
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
    G_z_spec = graph.get_tensor_by_name('G_z_spec:0')

    start = time.time()
    _G_z = sess.run(G_z, {z: _z})

    for i in range(ndisplay):
        # print('-' * 80)
        # print('Example {}'.format(i))
        # display(PIL.Image.fromarray(_G_z_spec[i]))
        # display(Audio(_G_z[i], rate=16000))
        librosa.output.write_wav('sample_audio/{}.wav'.format(i), _G_z[i], 16000) 
