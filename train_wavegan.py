import beatgan
import numpy as np
import sys

def train_wavegan(epochs, BATCH_SIZE):
    X_train = load_beat_data(1)
    np.random.shuffle(X_train)

    discriminator = get_discriminator()
    generator = get_generator()

    generator_model = make_generator_model(X_train, generator, discriminator)
    discriminator_model = make_discriminator_model(X_train, generator, discriminator)

    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
    for epoch in range(epochs):
        print("Epoch is", epoch)
        dl, gl = {}, {}
        np.random.shuffle(X_train)
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            audio_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE].reshape(BATCH_SIZE, 16384, hp.c)
            noise = get_noise((BATCH_SIZE, 100))
            d_loss = discriminator_model.train_on_batch([audio_batch, noise], [positive_y, negative_y, dummy_y])
            dl = d_loss
            if index % hp.D_updates_per_G_update == 0:
                #print("batch %d d_loss : %s" % (index, d_loss))
                noise = get_noise((BATCH_SIZE, 100))
                g_loss = generator_model.train_on_batch(noise, positive_y)
                gl = g_loss
                #print("batch %d g_loss : %0.10f" % (index, g_loss))

        if epoch % 500 == 0:
            print("epoch %d d_loss : %s" % (epoch, dl))
            print("epoch %d g_loss : %0.10f" % (epoch, gl))
            generator.save_weights('weights/generator' + str(epoch) + '.h5', True)
            discriminator.save_weights('weights/discriminator' + str(epoch) + '.h5', True)
            generate_one(generator, epoch, 0)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Please include number of epochs and batch size as arguments')
        sys.exit()
    train_wavegan(sys.argv[1], sys.argv[2])
