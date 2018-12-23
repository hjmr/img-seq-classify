import chainer
import chainer.links as L
import chainer.functions as F

import numpy
from chainer.backends import cuda

from config import Config as Cnf


class MySubNet(chainer.Chain):
    def __init__(self, in_ch, ch1, cv1, ch2, cv2):
        super(MySubNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_ch, ch1, cv1)
            self.conv2 = L.Convolution2D(ch1, ch2, cv2)

    def forward(self, x):
        h_1 = F.relu(self.conv1(x))
        h_2 = self.conv2(h_1)
        h_3 = F.max_pooling_2d(h_2, ksize=(4, 4))
        h_4 = F.local_response_normalization(h_3)
        return F.relu(h_4)


class MyNet(chainer.Chain):
    def __init__(self, image_num, gpuid=-1):
        img_channels = 1 if Cnf.IMAGE_MONO else 3
        super(MyNet, self).__init__()
        with self.init_scope():
            self.convNets = []
            for i in range(image_num):
                self.convNets.append(MySubNet(img_channels,
                                              Cnf.CONV1_OUT_CHANNELS, Cnf.CONV_SIZE,
                                              Cnf.CONV2_OUT_CHANNELS, Cnf.CONV_SIZE))
            self.fullLayer1 = L.Linear(in_size=None, out_size=Cnf.NUM_HIDDEN_NEURONS2)
            self.fullLayer2 = L.Linear(in_size=None, out_size=Cnf.NUM_CLASSES)
        self.gpuid = gpuid
        self.image_num = image_num

    def __call__(self, x, y_hat=None):
        if 0 <= self.gpuid:
            xp = cuda.cupy
        else:
            xp = numpy
        x_inp = xp.split(x, self.image_num, axis=2)
        h = []
        for i in range(self.image_num):
            h.append(self.convNets[i].forward(x_inp[i]))
        h_inp = xp.concatenate(h, axis=2)
        h_out = F.dropout(F.relu(self.fullLayer1(h_inp)))
        y = self.fullLayer2(h_out)
        if y_hat is None:
            return y
        else:
            loss = F.softmax_cross_entropy(y, y_hat)
            accu = F.accuracy(y, y_hat)
            chainer.report({'loss': loss, 'accuracy': accu}, observer=self)
            return loss
