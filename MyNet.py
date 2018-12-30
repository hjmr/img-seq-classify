import chainer
import chainer.links as L
import chainer.functions as F
from chainer.backends import cuda


CONV_FILTER_SIZE = 4
MAX_POOLING_SIZE = 3
NUM_HIDDEN_NEURONS = 2048


class MySubNet(chainer.Chain):
    def __init__(self):
        super(MySubNet, self).__init__()
        with self.init_scope():
            self.conv11 = L.Convolution2D(24, CONV_FILTER_SIZE)
            self.conv12 = L.Convolution2D(24, CONV_FILTER_SIZE)
            self.conv21 = L.Convolution2D(48, CONV_FILTER_SIZE)
            self.conv22 = L.Convolution2D(48, CONV_FILTER_SIZE)
            self.conv31 = L.Convolution2D(96, CONV_FILTER_SIZE)
            self.conv32 = L.Convolution2D(96, CONV_FILTER_SIZE)

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
        return h


class MyNet(chainer.Chain):
    def __init__(self, out_size, image_num, gpuid=-1):
        super(MyNet, self).__init__()
        with self.init_scope():
            self.convNets = []
            for i in range(image_num):
                self.convNets.append(MySubNet())
            self.fullLayer1 = L.Linear(in_size=None, out_size=NUM_HIDDEN_NEURONS)
            self.fullLayer2 = L.Linear(in_size=None, out_size=out_size)
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

    def forward(self, x):
        x_inp = F.split_axis(x, self.image_num, axis=2)
        h = [self.convNets[i].forward(x_inp[i]) for i in range(self.image_num)]
        h = F.concat(h, axis=2)
        h = F.dropout(F.relu(self.fullLayer1(h)))
        return self.fullLayer2(h)
