import argparse

import numpy as np
import chainer
from chainer import serializers
from chainer.backends import cuda

from MyNet import MyNet
from read_data import read_one_image

from PIL import Image


def parse_arg():
    parser = argparse.ArgumentParser(description='Test CNN')
    parser.add_argument('-c', '--out_class_num', type=int, default=4,
                        help='the number of output classes.')
    parser.add_argument('-n', '--image_num', type=int, default=5,
                        help='the number of images as a set of input.')
    parser.add_argument('-m', '--model_file', type=str, nargs=1,
                        help='load specified model file.')
    parser.add_argument('images', type=str, nargs='+',
                        help='images for testing')
    return parser.parse_args()


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def visualize_one_layer(layer_output):
    pad = 5
    filter_num = layer_output.shape[1]
    img_width = layer_output.shape[2]
    img_height = layer_output.shape[3]
    side = int(np.ceil(np.sqrt(filter_num)))
    canvas = np.zeros(((side + 1) * pad + img_height * side,
                       (side + 1) * pad + img_width * side), dtype=np.uint8)
    for idx in range(filter_num):
        o = layer_output[:, idx:idx+1, :, :].reshape((img_width, img_height))
        single_image = deprocess_image(o.data)
        x_pos = idx % side
        y_pos = idx // side
        canvas[(y_pos + 1) * pad + y_pos * img_height:
               (y_pos + 1) * pad + y_pos * img_height + img_height,
               (x_pos + 1) * pad + x_pos * img_width:
               (x_pos + 1) * pad + x_pos * img_width + img_width] = single_image
    return canvas


def calc_layer_output(args):
    model = MyNet(args.out_class_num, args.image_num)
    serializers.load_npz(args.model_file[0], model)

    test_images = []
    for f in args.images:
        test_images.append(read_one_image(f))
    test_data = np.concatenate(test_images[0:args.image_num], axis=1)

    h = None
    with chainer.configuration.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            h = model.calc_cnn_layer(np.array([test_data]))
    return h


if __name__ == '__main__':
    args = parse_arg()
    if args.image_num <= len(args.images):
        layer_output = calc_layer_output(args)

        for l in layer_output:
            img = visualize_one_layer(l)
            im = Image.fromarray(img)
            im.show()
            input("please press enter.")
