import argparse

import numpy
import chainer
from chainer import serializers
import chainer.functions as F
from chainer.backends import cuda

from MyNet import MyNet
from config import Config
from read_data import read_data


def parse_arg():
    parser = argparse.ArgumentParser(description='Test CNN')
    parser.add_argument('-n', '--image_num', type=int, default=5,
                        help='the number of images as a set of input.')
    parser.add_argument('-m', '--model_file', type=str, nargs='+',
                        help='load specified model file.')
    parser.add_argument('-t', '--test_file', type=str, nargs=1,
                        help='plain text file for testing')
    parser.add_argument('-g', '--gpuid', type=int, default=-1,
                        help='GPU ID for calculation.')
    return parser.parse_args()


def main():
    args = parse_arg()

    test_images, test_labels = read_data(args.test_file[0], args.image_num)
    model = MyNet(args.image_num)

    xp = numpy
    if 0 <= args.gpuid:
        cuda.get_device_from_id(args.gpuid).use()
        model.to_gpu()
        xp = cuda.cupy

    results = []
    for m in args.model_file:
        print("# testing for {} ... ".format(m))
        serializers.load_npz(m, model)

        with chainer.configuration.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                for i in range(len(test_images)):
                    y = model(xp.array([test_images[i]]))
                    pred = F.argmax(y)
                    if test_labels is not None:
                        results.append((pred.data, test_labels[i]))
                    else:
                        print(pred.data)

    # count TP,TF,FP,FF
    if test_labels is not None:
        for c in range(Config.NUM_CLASSES):
            v = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
            for p, t in results:
                if t == c and p == c:
                    v['TP'] += 1
                elif t == c:
                    v['FN'] += 1
                elif p == c:
                    v['FP'] += 1
                else:
                    v['TN'] += 1
            accuracy = (v['TP'] + v['TN']) / len(results)
            precision = v['TP'] / (v['TP'] + v['FP']) if 0 < (v['TP'] + v['FP']) else 0
            recall_rate = v['TP'] / (v['TP'] + v['FN']) if 0 < (v['TP'] + v['FN']) else 0
            F_measure = 2 * recall_rate * precision / (recall_rate + precision) if 0 < (recall_rate + precision) else 0
            print('class:{}, accuracy:{}, precision:{}, recall_rate:{}, F_measure:{}'.format(
                c, accuracy, precision, recall_rate, F_measure))


if __name__ == '__main__':
    main()
