import argparse
import params as pp
import nn
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
import settings as s
import os


def train(data, layers, restore, epochs, save, keepprob=1., **kwargs):
    """ train the CNN """
    cnn = nn.CNN(layers, low_mem=True, save=save, **kwargs)
    if restore is not None:
        cnn.restore(restore)
    print 'training model %s' % cnn.name
    cnn.train(data, epochs=epochs, keepprob=keepprob)
    print 'saved model to %s/%s' % (s.MODELS_DIR, cnn.name)
    print 'writing results to %s/%s' % (s.RESULTS_DIR, cnn.name)
    if save:
        cnn.save_model()
    results = get_results(data, cnn, test=False)
    summarize_results(results, '%s_train.txt' % cnn.name)
    return cnn

def get_results(data, cnn, test=True):
    results = []
    for i in range(3, 9):
        if test:
            X, y = data.get_test(i)
        else:
            X, y = data.get_train(i)
        yhat = cnn.test(X)
        print(y.shape, X[:,:,4].sum(axis=1).shape)
        np.testing.assert_equal(X[:,:,4].sum(axis=1), X[:,:,5].sum(axis=1))
        results.append(pd.DataFrame({'dG_true': y[:,0],
                                     'dH_true': y[:,1],
                                     'dG_predicted': yhat[:,0],
                                     'dH_predicted': yhat[:,1],
                                     'length': i,
                                     'num_pairs': X[:,:,4].sum(axis=1)}))
    return pd.concat(results)

def summarize_results(results, filename, scale=1):
    r_dG = stats.pearsonr(results.dG_true, results.dG_predicted)
    r_dH = stats.pearsonr(results.dH_true, results.dH_predicted)
    rmse_dG = np.sqrt(np.mean(np.square(results.dG_true - results.dG_predicted)))
    rmse_dH = np.sqrt(np.mean(np.square(results.dH_true - results.dH_predicted)))
    print 'dG: r = %.4f, p = %.4e, rmse = %.4d' % (r_dG[0], r_dG[1], rmse_dG)
    print 'dH: r = %.4f, p = %.4e, rmse = %.4d' % (r_dH[0], r_dH[1], rmse_dH)
    if scale != 1:
        scaled_rmse = rmse_dG * scale + rmse_dH 
        print 'scaled rmse = ' % scaled_rmse
    results.to_csv('%s/%s' % (s.RESULTS_DIR, filename), sep='\t', index=False)

def test(data, cnn, scale=1):
    """ test the CNN and print results to file """
    results = get_results(data, cnn, test=True)
    summarize_results(results, '%s_test.txt' % cnn.name)


def parse_args():
    parser = argparse.ArgumentParser(description='train and test a CNN for '
                                                 'nearest neighbor parameters')
    parser.add_argument('filename', help='name of parameter file')
    parser.add_argument('n_units', help='numbers of units in hidden layers')
    parser.add_argument('-o', '--optimizer', help='optimizer type (either "descent", '
                        '"adam", "adagrad", or "rmsprop")', default='rmsprop')
    parser.add_argument('-l', '--learning_rate', help='learning rate for '
                        'model training', type=float, default=1e-4)
    parser.add_argument('-e', '--epochs', help='number of training epochs',
                        type=int, default=10000)
    parser.add_argument('-p', '--keepprob', help='probability to keep a node '
                        'in dropout', type=float, default=1.)
    parser.add_argument('-r', '--restore', help='file to restore weights from',
                        default=None)
    parser.add_argument('-s', '--save', help='save variables',
                        action='store_true')
    parser.add_argument('-c', '--scale', help='scale factor between dG and dH',
                        default=1, type=float)
    parser.add_argument('-d', '--seed', help='random seed', type=int)
    parser.add_argument('-n', '--norm', help='order of norm for loss function',
                        type=int, default=1)
    parser.add_argument('--dilate', help='use dilated convlutions',
                        action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    paramfile = os.path.splitext(os.path.basename(args.filename))[0]
    name = '%s_%slayer_%s%.0e' % \
        (paramfile, args.n_units, args.optimizer, args.learning_rate)

    if paramfile.startswith('dna'):
        s.set_molecule('dna')
    elif paramfile.startswith('rna'):
        s.set_molecule('rna')

    params = pp.NNParams(args.filename, split=True)
    cnn = train(params, args.n_units, args.restore, args.epochs, save=args.save,
                optimizer=args.optimizer, learning_rate=args.learning_rate,
                name=name, scale=args.scale, norm=args.norm, dilate=args.dilate,
                keepprob=args.keepprob)
    test(params, cnn, args.scale)


if __name__ == '__main__':
    main()
