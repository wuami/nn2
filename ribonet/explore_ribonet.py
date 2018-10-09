import tensorflow as tf
import nn
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='explore neuron activations'
                                                 'for a nearest neighbor CNN')
    parser.add_argument('-f', '--filename', help='name of model file')
    parser.add_argument('-c', '--combine_layers', help='turn off combining'
                        'of all layers into fully connected layer',
                        action='store_false')
    parser.add_argument('-l', '--layers', help='sizes of hidden layers',
                        metavar='N', type=int, nargs='+')
    return parser.parse_args()


def main():
    args = parse_args()
    cnn = nn.CNN(args.layers, write=True)
    cnn.restore(args.filename)
    # cnn.get_activation_profile(1,0, iters=1000, learning_rate=0.1)
    for i, layer in enumerate(args.layers):
        for j in range(layer):
            cnn.get_activation_profile(i+1, j)

if __name__ == '__main__':
    main()
