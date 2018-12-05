import beatgan
import sys

def generate_batch(generator, weights_file, pickle_file):
    _, source_videos = beatgan.load_videos(pickle_file, beatgan.hp.window_radius, beatgan.hp.downsample_factor)
    generator.load_weights(weights_file)
    generated_audio = generator.predict(source_videos, verbose=1)
    re_normalization_factor = 8388608
    assumed_sample_length = 14112
    sample_rate = 14700
    for i in range(len(generated_audio)):
        output = generated_audio[i]
        q = np.array(output*re_normalization_factor).astype('int32')
        wavfile24.write('generated_outputs/output' + ("%03d" % i) + '.wav', sample_rate, np.concatenate((q[:assumed_sample_length], q[:assumed_sample_length])), bitrate=24)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Please provide a weights file to generate from and a pickle file of source videos')
        sys.exit()
    weights_file = sys.argv[1]
    pickle_file = sys.argv[2]
    generator = beatgan.get_generator()
    generate_batch(generator, weights_file, pickle_file)
