import argparse

import numpy as np
from PIL import Image

from config import Config


def normalize_image(img):
    # モノクロ化（convert）
    if Config.IMAGE_MONO:
        img = img.convert('L')
    # リサイズ
    img = img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.ANTIALIAS)
    # np.array形式に変換
    img = np.array(img, dtype=np.float32)
    # (channel, height, width）の形式に変換
    if Config.IMAGE_MONO:
        img = img[np.newaxis, :]
    else:
        img = img.transpose([2, 0, 1])
    # 0-1のfloat値にする
    img /= 255.0
    return img


def read_one_image(filename):
    return normalize_image(Image.open(filename))


def read_data(filename, image_num):
    # データを入れる配列
    images = []
    labels = []
    # ファイルを開く
    with open(filename, 'r') as f:
        for line in f:
            # 改行を除いてスペース区切りにする
            words = line.rstrip().split()
            # イメージを読み込み
            images.append(read_one_image(words[0]))
            if 2 <= len(words):
                # 対応するラベルを用意
                labels.append(int(words[1]))

    input_data = []
    teach_data = []
    for i in range(len(images) - (image_num - 1)):
        input_data.append(np.concatenate(images[i:i + image_num], axis=1))
        if 0 < len(labels):
            teach_data.append(labels[i + image_num // 2])
    if len(teach_data) == 0:
        teach_data = None
    return input_data, teach_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Read Data')
    parser.add_argument('-n', '--image_num', type=int, default=5,
                        help='the number of images as a set of input.')
    parser.add_argument('data_file', type=str, nargs=1,
                        help='a text file which contains image file name and corresponding label in each line.')
    args = parser.parse_args()
    images, labels = read_data(args.data_file[0], args.image_num)
    print(labels)
