import argparse

import numpy as np
import chainer
from chainer import serializers
import chainer.functions as F

from MyNet import MyNet
from config import Config
from read_data import read_data


def parse_arg():
    parser = argparse.ArgumentParser(description='Test CNN')
    parser.add_argument('-n', '--image_num', type=int, default=5,
                        help='the number of images as a set of input.')
    parser.add_argument('-m', '--model_file', type=str, nargs='+',
                        help='load specified model file.')
    parser.add_argument('test_file', type=str, nargs=1,
                        help='plain text file for testing')
    return parser.parse_args()


def main():
    args = parse_arg()

    test_images, test_labels = read_data(args.test_file[0], args.image_num)
    model = MyNet(args.image_num)

    results = []
    for m in args.model_file:
        serializers.load_npz(m, model)

        with chainer.using_config("train", False):
            for i in range(len(test_images)):
                y = model(np.array([test_images[i]]))
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
            precision = v['TP'] / (v['TP'] + v['FP'])
            recall_rate = v['TP'] / (v['TP'] + v['FN'])
            F_measure = 2 * recall_rate * precision / (recall_rate + precision)
            print('class:{}, accuracy:{}, precision:{}, recall_rate:{}, F_measure:{}'.format(
                c, accuracy, precision, recall_rate, F_measure))


if __name__ == '__main__':
    main()
