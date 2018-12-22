import beatgan
import numpy as np

wavegan_instance = beatgan.get_wavegan()
generator = beatgan.get_generator(wavegan_instance)
discriminator = beatgan.get_discriminator()

beatgan.train(2000)
