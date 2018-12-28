import chainer
import chainer.links as L
import chainer.functions as F

from chainer.backends import cuda

from config import Config as Cnf


class MySubNet(chainer.Chain):
    def __init__(self, in_ch, ch1, cv1, ch2, cv2, ch3, cv3):
        super(MySubNet, self).__init__()
        with self.init_scope():
            self.conv11 = L.Convolution2D(in_ch, ch1, cv1)
            self.conv12 = L.Convolution2D(ch1, ch1, cv1)
            self.conv21 = L.Convolution2D(ch1, ch2, cv2)
            self.conv22 = L.Convolution2D(ch2, ch2, cv2)
            self.conv31 = L.Convolution2D(ch2, ch3, cv3)
            self.conv32 = L.Convolution2D(ch3, ch3, cv3)

    def forward(self, x):
        h = F.relu(self.conv11(x))
        h = F.relu(self.conv12(h))
        h = F.max_pooling_2d(h, ksize=(4, 4))
        h = F.local_response_normalization(h)
        h = F.relu(self.conv21(h))
        h = F.relu(self.conv22(h))
        h = F.max_pooling_2d(h, ksize=(4, 4))
        h = F.local_response_normalization(h)
        h = F.relu(self.conv31(h))
        h = F.relu(self.conv32(h))
        h = F.max_pooling_2d(h, ksize=(4, 4))
        h = F.local_response_normalization(h)
        return h


class MyNet(chainer.Chain):
    def __init__(self, image_num, gpuid=-1):
        img_channels = 1 if Cnf.IMAGE_MONO else 3
        super(MyNet, self).__init__()
        with self.init_scope():
            self.convNets = []
            for i in range(image_num):
                self.convNets.append(MySubNet(img_channels,
                                              Cnf.CONV1_OUT_CHANNELS, Cnf.CONV_SIZE,
                                              Cnf.CONV2_OUT_CHANNELS, Cnf.CONV_SIZE,
                                              Cnf.CONV3_OUT_CHANNELS, Cnf.CONV_SIZE))
            self.fullLayer1 = L.Linear(in_size=None, out_size=Cnf.NUM_HIDDEN_NEURONS2)
            self.fullLayer2 = L.Linear(in_size=None, out_size=Cnf.NUM_CLASSES)
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

    # def __call__(self, x, y_hat=None):
    #     y = self.forward(x)
    #     if y_hat is None:
    #         return y
    #     else:
    #         loss = F.softmax_cross_entropy(y, y_hat)
    #         accu = F.accuracy(y, y_hat)
    #         chainer.report({'loss': loss, 'accuracy': accu}, observer=self)
    #         return loss
