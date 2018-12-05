import beatgan
import sys

def generate_batch(generator, weights_file, pickle_file):
    _, source_videos = beatgan.load_videos(pickle_file, beatgan.hp.window_radius)
    generator.load_weights(weights_file)
    generated_audio = generator.predict(source_videos, verbose=1)
    print(generated_audio)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Please provide a weights file to generate from and a pickle file of source videos')
        sys.exit()
    weights_file = sys.argv[1]
    pickle_file = sys.argv[2]
    generator = beatgan.get_generator()
    generate_batch(generator, weights_file, pickle_file)
