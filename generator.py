from beatgan import get_generator
import sys

def generate_batch(generator, weights_file):
    noise = get_noise((hp.b,100))
    generator.load_weights(weights_file)
    generated_audio = generator.predict(noise, verbose=1)
    re_normalization_factor = 8388608
    assumed_sample_length = 14112
    sample_rate = 14700
    for i in range(len(generated_audio)):
        output = generated_audio[i]
        q = np.array(output*re_normalization_factor).astype('int32')
        wavfile24.write('generated_outputs/output' + ("%03d" % i) + '.wav', sample_rate, np.concatenate((q[:assumed_sample_length], q[:assumed_sample_length])), bitrate=24)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Please provide a weights file to generate from')
        sys.exit()
    weights_file = sys.argv[1]
    generator = get_generator()
    generate_batch(generator, weights_file)
