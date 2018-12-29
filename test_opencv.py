import argparse

import numpy as np
import chainer
from chainer import serializers
import chainer.functions as F
import chainer.links as L

import cv2
from PIL import Image

from MyNet import MyNet
from read_data import normalize_image


def parse_arg():
    parser = argparse.ArgumentParser(description='Test CNN')
    parser.add_argument('-n', '--image_num', type=int, default=5,
                        help='the number of images as a set of input.')
    parser.add_argument('-m', '--model_file', type=str, nargs=1,
                        help='load specified model file.')
    return parser.parse_args()


def main():
    args = parse_arg()

    model = L.Classifier(MyNet(args.image_num))
    serializers.load_npz(args.model_file[0], model)

    vc = cv2.VideoCapture(0)

    images = []

    with chainer.configuration.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            while True:
                rval, cv_img = vc.read()
                if rval:
                    cv2.imshow("preview", cv_img)
                    img = normalize_image(Image.fromarray(cv_img[::-1, :, ::-1]))
                    images.append(img)
                    while args.image_num < len(images):
                        images.pop(0)

                    if len(images) == args.image_num:
                        y = model.predictor.forward(np.array([np.concatenate(images)]))
                        pred = F.argmax(y)
                        print(pred.data)

                else:
                    print("None...")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == '__main__':
    main()
