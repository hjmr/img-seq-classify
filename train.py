import argparse

import chainer
import chainer.links as L
from chainer import optimizers, serializers, iterators, training
from chainer.backends import cuda, intel64
from chainer.training import extensions

from config import Config
from MyNet import MyNet
from read_data import read_data


def parse_arg():
    parser = argparse.ArgumentParser(description='Train CNN')
    parser.add_argument('-n', '--image_num', type=int, default=5,
                        help='the number of images as a set of input.')
    parser.add_argument('-e', '--epoch', type=int, default=100,
                        help='the training epochs.')
    parser.add_argument('-b', '--batch_size', type=int, default=20,
                        help='batch size.')
    parser.add_argument('-s', '--save_interval', type=int, default=100,
                        help='save models at every specified interval of the epoch.')
    parser.add_argument('-g', '--gpuid', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('-o', '--output_dir', type=str, default='.',
                        help='all output will be stored in the specified output directory.')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='plot loss and accuracy.')
    parser.add_argument('-r', '--resume_from_snapshot', type=str,
                        help='resume training from specified snapshot.')
    parser.add_argument('--arrange_data', action='store_true',
                        help='arrange data to have the same number of data in each class.')
    parser.add_argument('--ideep', action='store_true',
                        help='use iDeep to accelerate computation.')
    parser.add_argument('train_data', type=str, nargs=1,
                        help='plain text file for training.')
    parser.add_argument('test_data', type=str, nargs='?',
                        help='plain text file for testing')
    return parser.parse_args()


def arrange_data(all_data, all_labels):
    data_of_class = {i: [] for i in range(Config.NUM_CLASSES)}
    for d, l in zip(all_data, all_labels):
        data_of_class[l].append(d)

    max_len = max([len(d) for d in data_of_class.values()])

    data = []
    labels = []
    for i in range(Config.NUM_CLASSES):
        d = data_of_class[i]
        data.extend([d[j % len(d)] for j in range(max_len)])
        labels.extend([i] * max_len)
    return data, labels


def main():
    args = parse_arg()
    if 0 <= args.gpuid and args.ideep:
        print('GPU and iDeep cannot use simultaneously.')
        return

    model = L.Classifier(MyNet(args.image_num, gpuid=args.gpuid))
    if 0 <= args.gpuid:
        cuda.get_device_from_id(args.gpuid).use()
        model.to_gpu()

    if args.ideep and intel64.is_ideep_available():
        chainer.global_config.use_ideep = 'auto'
        model.to_intel64()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    if args.test_data:
        train_images, train_labels = read_data(args.train_data[0], args.image_num)
        test_images, test_labels = read_data(args.test_data, args.image_num)
        if args.arrange_data:
            train_images, train_labels = arrange_data(train_images, train_labels)
            test_images, test_labels = arrange_data(test_images, test_labels)
        train_data = chainer.datasets.TupleDataset(train_images, train_labels)
        test_data = chainer.datasets.TupleDataset(test_images, test_labels)
    else:
        data_images, data_labels = read_data(args.train_data[0], args.image_num)
        data_tuples = chainer.datasets.TupleDataset(data_images, data_labels)
        train_size = int(len(data_tuples) * 0.8)
        train_data, test_data = chainer.datasets.split_dataset_random(data_tuples, train_size)

    train_iter = iterators.SerialIterator(train_data, args.batch_size)
    test_iter = iterators.SerialIterator(test_data, args.batch_size, repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpuid)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.output_dir)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpuid))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'), trigger=(args.save_interval, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='model_epoch-{.updater.epoch}'), trigger=(args.save_interval, 'epoch'))
    trainer.extend(extensions.LogReport())

    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    if args.resume_from_snapshot is not None:
        serializers.load_npz(args.resume_from_snapshot, trainer)

    trainer.run()

    outfile = "{}/mynet.model".format(args.output_dir)
    serializers.save_npz(outfile, model.predictor)


if __name__ == '__main__':
    main()
