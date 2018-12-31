import chainer
import chainer.links as L
import chainer.functions as F
from chainer.backends import cuda


CONV_FILTER_SIZE = 3
MAX_POOLING_SIZE = 2

NUM_HIDDEN_NEURONS1 = 4096
NUM_HIDDEN_NEURONS2 = 1024


class MySubNet(chainer.Chain):
    def __init__(self):
        super(MySubNet, self).__init__()
        with self.init_scope():
            self.conv11 = L.Convolution2D(32, CONV_FILTER_SIZE)
            self.conv12 = L.Convolution2D(32, CONV_FILTER_SIZE)
            self.conv21 = L.Convolution2D(48, CONV_FILTER_SIZE)
            self.conv22 = L.Convolution2D(48, CONV_FILTER_SIZE)
            self.conv31 = L.Convolution2D(64, CONV_FILTER_SIZE)
            self.conv32 = L.Convolution2D(64, CONV_FILTER_SIZE)
            self.conv41 = L.Convolution2D(96, CONV_FILTER_SIZE)
            self.conv42 = L.Convolution2D(96, CONV_FILTER_SIZE)

    def forward(self, x):
        h = F.relu(self.conv11(x))
        h = F.relu(self.conv12(h))
        h = F.max_pooling_2d(h, ksize=MAX_POOLING_SIZE)
        h = F.relu(self.conv21(h))
        h = F.relu(self.conv22(h))
        h = F.max_pooling_2d(h, ksize=MAX_POOLING_SIZE)
        h = F.relu(self.conv31(h))
        h = F.relu(self.conv32(h))
        h = F.max_pooling_2d(h, ksize=MAX_POOLING_SIZE)
        h = F.relu(self.conv41(h))
        h = F.relu(self.conv42(h))
        h = F.max_pooling_2d(h, ksize=MAX_POOLING_SIZE)
        return h


class MyNet(chainer.Chain):
    def __init__(self, out_size, image_num, gpuid=-1):
        super(MyNet, self).__init__()
        with self.init_scope():
            self.convNets = []
            for i in range(image_num):
                self.convNets.append(MySubNet())
            self.fullLayer1 = L.Linear(in_size=None, out_size=NUM_HIDDEN_NEURONS1)
            self.fullLayer2 = L.Linear(in_size=None, out_size=NUM_HIDDEN_NEURONS2)
            self.fullLayer3 = L.Linear(in_size=None, out_size=out_size)
        self.gpuid = gpuid
        self.image_num = image_num

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            super(MyNet, self).to_gpu()
            for net in self.convNets:
                net.to_gpu()
        return self

    def to_intel64(self):
        super(MyNet, self).to_intel64()
        for net in self.convNets:
            net.to_intel64()

    def calc_cnn_layer(self, x):
        x_inp = F.split_axis(x, self.image_num, axis=2)
        h = [self.convNets[i].forward(x_inp[i]) for i in range(self.image_num)]
        return h

    def calc_ffnn_layer(self, x):
        h = F.dropout(F.relu(self.fullLayer1(x)))
        h = F.dropout(F.relu(self.fullLayer2(h)))
        return self.fullLayer3(h)

    def forward(self, x):
        h = self.calc_cnn_layer(x)
        h = F.concat(h, axis=2)
        return self.calc_ffnn_layer(h)
