import sys
import argparse

import numpy as np
import chainer
from chainer import serializers

from MyNet import MyNet
from read_data import read_one_image


def parse_arg():
    parser = argparse.ArgumentParser(description='Test CNN')
    parser.add_argument('-n', '--image_num', type=int, default=5,
                        help='the number of images as a set of input.')
    parser.add_argument('-m', '--model_file', type=str, nargs=1,
                        help='load specified model file.')
    parser.add_argument('images', type=str, nargs='+',
                        help='images for testing')
    return parser.parse_args()


def main():
    args = parse_arg()

    model = MyNet(args.image_num)
    serializers.load_npz(args.model_file[0], model)

    test_images = []
    for f in args.images:
        test_images.append(read_one_image(f))

    test_data = []
    for i in range(len(test_images) - (args.image_num - 1)):
        test_data.append(np.concatenate(test_images[i:i + args.image_num]))

    with chainer.configuration.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            y = model.forward(test_data)
            pred = np.argmax(y)
            print(pred)


if __name__ == '__main__':
    main()
