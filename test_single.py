import sys
import argparse

import numpy
import chainer
from chainer import serializers
import chainer.functions as F
from chainer.backends import cuda

from MyNet import MyNet
from read_data import read_one_image


def parse_arg():
    parser = argparse.ArgumentParser(description='Test CNN')
    parser.add_argument('-c', '--out_class_num', type=int, default=4,
                        help='the number of output classes.')
    parser.add_argument('-n', '--image_num', type=int, default=5,
                        help='the number of images as a set of input.')
    parser.add_argument('-m', '--model_file', type=str, nargs=1,
                        help='load specified model file.')
    parser.add_argument('-g', '--gpuid', type=int, default=-1,
                        help='GPU ID for calculation.')
    parser.add_argument('images', type=str, nargs='+',
                        help='images for testing')
    return parser.parse_args()


def main():
    args = parse_arg()

    model = MyNet(args.out_class_num, args.image_num)
    xp = numpy
    if 0 <= args.gpuid:
        cuda.get_device_from_id(args.gpuid).use()
        model.to_gpu()
        xp = cuda.cupy

    serializers.load_npz(args.model_file[0], model)

    test_images = []
    for f in args.images:
        test_images.append(read_one_image(f))

    test_data = []
    for i in range(len(test_images) - (args.image_num - 1)):
        test_data.append(numpy.concatenate(test_images[i:i + args.image_num]))

    with chainer.configuration.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            for i in range(len(test_data)):
                y = model.forward(xp.array([test_data[i]]))
                pred = F.argmax(y)
                print(pred.data)


if __name__ == '__main__':
    main()
