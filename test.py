import argparse

import numpy as np
import chainer
from chainer import serializers
import chainer.functions as F

from MyNet import MyNet
from read_data import read_test_data


def parse_arg():
    parser = argparse.ArgumentParser(description='Test CNN')
    parser.add_argument('-n', '--image_num', type=int, default=5,
                        help='the number of images as a set of input.')
    parser.add_argument('-m', '--model_file', type=str, nargs=1,
                        help='load specified model file.')
    parser.add_argument('test_file', type=str, nargs=1,
                        help='plain text file for testing')
    return parser.parse_args()


def main():
    args = parse_arg()

    model = MyNet(args.image_num)
    serializers.load_npz(args.model_file[0], model)

    test_images = read_test_data(args.test_file[0], args.image_num)

    with chainer.using_config("train", False):
        for t in test_images:
            y = model(np.array([t]))
            pred = F.argmax(y)
            print(pred.data)


if __name__ == '__main__':
    main()
