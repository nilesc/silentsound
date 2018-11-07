from wavegan.wavegan import WaveGANGenerator
import tensorflow as tf
import numpy as np
import PIL.Image
from IPython.display import display, Audio
import time as time

if __name__ == '__main__':
    # Load the model
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model.ckpt')

    # Generate and display audio
    
    # CHANGE THESE to change number of examples generated/displayed
    ngenerate = 64
    ndisplay = 4
    
    # Sample latent vectors
    _z = (np.random.rand(ngenerate, 100) * 2.) - 1.
    
    # Generate
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
    G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
    
    start = time.time()
    _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: _z})
    print('Finished! (Took {} seconds)'.format(time.time() - start))

    for i in range(ndisplay):
      print('-' * 80)
      print('Example {}'.format(i))
      display(PIL.Image.fromarray(_G_z_spec[i]))
      display(Audio(_G_z[i], rate=16000))
