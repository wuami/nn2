import argparse
import ensemble as e
import params as pp
import model_utils as m
import settings as s
import tensorflow as tf
import pandas as pd
import numpy as np
import os, sys
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='train a CNN for '
                                                 'nearest neighbor parameters')
    parser.add_argument('filename', help='name of parameter file')
    parser.add_argument('n_units', help='numbers of units in hidden '
                        'layers')
    parser.add_argument('-o', '--optimizer', help='optimizer type (either '
                        '"descent", "adam", "adagrad", or "rmsprop")',
                        default='rmsprop')
    parser.add_argument('-e', '--epochs', help='number of epochs to train',
                        type=int, default=100)
    parser.add_argument('-b', '--batch_size', help='number of data points in '
                        'each training batch', type=int, default=10)
    parser.add_argument('-p', '--keepprob', help='probability to keep a node '
                        'in dropout', type=float, default=1.)
    parser.add_argument('-z', '--regularize', help='regularization parameter '
                        'for motifs', type=float, default=0)
    parser.add_argument('-l', '--learning_rate', help='learning rate for '
                        'model training', type=float, default=1e-4)
    parser.add_argument('-m', '--low_mem', help='low memory mode',
                        action='store_false')
    parser.add_argument('--batch_norm', help='enable batch '
                        'normalization', action='store_true')
    parser.add_argument('-r', '--restore', help='file to restore weights from')
    parser.add_argument('-s', '--save', help='save log files and models',
                        action='store_true')
    parser.add_argument('--sterr', help='whether not to use standard error in '
                        'loss function', action='store_true')
    parser.add_argument('-g', '--num_gpus', help='number of gpus', type=int,
                        default=1)
    parser.add_argument('-t', '--testfile', help='test dataset')
    parser.add_argument('-c', '--scale', help='scale factor between dG and dH',
                        default=1, type=float)
    parser.add_argument('-n', '--norm', help='order of norm for loss function',
                        type=int, default=1)
    parser.add_argument('-d', '--seed', help='random seed', type=int)
    parser.add_argument('-a', '--alpha', help='regularization parameter for '
                        'weights', type=float, default=0)
    parser.add_argument('--linear', help='linear least squares fit',
                        action='store_true')
    parser.add_argument('--dilate', help='use dilated convlutions',
                        action='store_true')
    return parser.parse_args()


def print_flush(s):
    """ print a string and flush buffer """
    print s
    sys.stdout.flush()


def strip_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_file(filename):
    if filename.endswith('kd'):
        return e.KdDataset([filename])
    elif filename.endswith('melt'):
        return e.MeltDataset([filename])
    elif filename.endswith('dG'):
        return e.dGDataset(filename)
    else:
        raise ValueError('filename must have suffix .kd or .melt '
                         'specifying the type of dataset')


def main():
    args = parse_args()
    name = '%s_%slayer_%.0e' % (strip_path(args.filename), args.n_units,
                                args.learning_rate)

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # determine if data is being restored from file
    print_flush('reading dataset')
    restore_data = False
    if args.restore is not None:
        basename = args.restore.rsplit('-', 1)[0]
        if os.path.isfile('%s.dat' % basename):
            restore_data = True
    if not restore_data:
        data = read_file(args.filename)
    else:
        data = None
    if args.testfile:
        testdata = read_file(args.testfile)
    else:
        testdata = None

    if args.linear:
        if not args.filename.endswith('dG'):
            print_flush('linear fit not yet available for melt and Kd data')
        p = pp.NNParams('%s/resources/nupack/parameters/%s' % (s.BASE_DIR,
                                                               s.paramfile))
        motifs, b_dG, b_dH = data.get_linear_fit()
        d = pd.DataFrame({'motif': motifs, 'fit_dG': b_dG, 'fit_dH': b_dH})
        nupack_params = d.motif.apply(p.get_energy)
        d['nupack_dG'] = [x[0] for x in nupack_params]
        d['nupack_dH'] = [x[1] for x in nupack_params]
        filename = '%s/linear_fit_%s' % (s.RESULTS_DIR, e.rand_string())
        print_flush('writing results to %s' % filename)
        d.to_csv('%s.params' % filename, sep='\t', index=False)
        data.to_file('%s.txt' % filename)
        return

    print_flush('initializing model')
    if args.filename.endswith('dG'):
        cnn = m.MeltTrainer(data, args.n_units, save=args.save,
                            learning_rate=args.learning_rate,
                            name=name, optimizer=args.optimizer,
                            num_gpus=args.num_gpus, scale=args.scale,
                            norm=args.norm, dilate=args.dilate,
                            testdata=testdata, alpha=args.alpha)
    else:
        cnn = m.RiboNetTrainer(data, args.n_units, save=args.save,
                               learning_rate=args.learning_rate,
                               batch_norm=args.batch_norm, name=name,
                               low_mem=args.low_mem, optimizer=args.optimizer,
                               num_gpus=args.num_gpus,
                               norm=args.norm, dilate=args.dilate,
                               testdata=testdata, alpha=args.alpha)

    if args.restore is not None:
        print_flush('restoring model weights')
        cnn.restore(args.restore)

    print_flush('training model %s' % cnn.name)
    loss = cnn.train(args.epochs, args.batch_size, args.keepprob,
                     alpha=args.regularize, sterr=args.sterr)
    finalloss = cnn.get_loss(sterr=args.sterr)
    print 'final train loss: %s' % finalloss
    print_flush('writing results to %s/%s' % (s.RESULTS_DIR, cnn.name))
    if args.save:
        cnn.save_model('_%.3f' % finalloss.sum())
    cnn.write_results_file('%s/%s.txt' % (s.RESULTS_DIR, cnn.name))
    cnn.params.to_file('%s/%s' % (s.RESULTS_DIR, cnn.name))
    np.savetxt('%s/%s.loss' % (s.RESULTS_DIR, cnn.name), loss, delimiter='\t',
               header='train\ttest', comments='')
    
    if args.testfile is not None:
        data = read_file(args.testfile)
        cnn.update_dataset(data)
        print 'test loss: %s' % cnn.get_loss(sterr=args.sterr)
        print 'test r2: %s' % cnn.get_r2()
        cnn.write_results_file('%s/%s_%s.txt' % (s.RESULTS_DIR, cnn.name,
                                                 strip_path(args.testfile)))


if __name__ == '__main__':
    main()
