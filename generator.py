from wavegan.wavegan import WaveGANGenerator

class Generator:

    def __init__(self):
        """ Initializes the generator. Currently a placeholder. """    
        print("Hello, world!")

if __name__ == '__main__':
    # Load the model
    import tensorflow as tf
    
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model.ckpt')

    my_generator = Generator()
