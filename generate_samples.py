import sys
from beatgan import generate_batch, get_generator

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Provide the path to a weights file as a command line argument')
    generator = get_generator()
    generate_batch(generator, sys.argv[1])
