import beatgan
import numpy as np

generator = beatgan.get_generator()
discriminator = beatgan.get_discriminator()

beatgan.train(50)
