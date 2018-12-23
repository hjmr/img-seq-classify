import argparse

import chainer
from chainer import optimizers, serializers, iterators, training
from chainer.backends import cuda
from chainer.training import extensions

from config import Config
from MyNet import MyNet
from read_data import read_train_data


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
    parser.add_argument('data_files', type=str, nargs=2,
                        help='plain text file for training and testing')
    return parser.parse_args()


def main():
    args = parse_arg()

    model = MyNet(args.image_num, gpuid=args.gpuid)
    if 0 <= args.gpuid:
        cuda.get_device_from_id(args.gpuid).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train_images, train_labels = read_train_data(args.data_files[0], args.image_num)
    train_data = chainer.datasets.TupleDataset(train_images, train_labels)
    train_iter = iterators.SerialIterator(train_data, args.batch_size)

    test_images, test_labels = read_train_data(args.data_files[1], args.image_num)
    test_data = chainer.datasets.TupleDataset(test_images, test_labels)
    test_iter = iterators.SerialIterator(test_data, args.batch_size, repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpuid)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.output_dir)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpuid))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.save_interval, 'epoch'))
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
    serializers.save_npz(outfile, model)


if __name__ == '__main__':
    main()
