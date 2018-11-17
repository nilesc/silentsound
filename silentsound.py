import beatgan

# generator = beatgan.get_generator(64, 2)
# discriminator = beatgan.get_discriminator(64, 2)
# print(beatgan.generator_containing_discriminator(generator, discriminator))

beatgan.train(1, 10)
