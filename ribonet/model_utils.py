import nn as n
import settings as s
import params as pp
import ensemble as e
import numpy as np
import random
import string
import os
import sys
import multiprocessing
import multiprocessing.pool
import time
import pickle
import warnings


class NoDaemonProcess(multiprocessing.Process):
    """ process with daemon set to false """
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class Pool(multiprocessing.pool.Pool):
    """ pool with no daemon process """
    Process = NoDaemonProcess


def run_in_child_process(func, args):
    """ run function in child process """
    p = Pool()
    res = p.apply_async(func, args)
    p.close()
    p.join()
    return res.get()


def get_train_batch(data, params, batch_size, i, sterr=True):
    """ get one batch of data for training """
    data.copy = True

    # pick batch of measurements and length
    np.random.seed(int(time.time()) + i)
    ls = {}
    while len(ls) == 0:
        r = np.random.choice(data.n, batch_size, replace=False)
        data.unpickle_ensembles(r)
        ls = data.get_motif_lengths(r)
    l = np.random.choice(ls.keys(), 1, p=ls.values())[0]

    # get proper motif shape
    motifs = data.get_motif_array(l, r)
    Ts = np.expand_dims(data.get_temps_motifs(r, l), 1).astype(float)
    # batch size also used to limit number of motifs used
    if motifs.shape[0] > batch_size:
        select = np.random.choice(motifs.shape[0], batch_size, replace=False)
        motifs = motifs[select, :, :]
        Ts = Ts[select]

    # get gradients
    data.update_energies(params, r)
    grad = data.get_grad_losses(motifs, r, sterr)
    data.clear_ensembles(r)
    return r, motifs, grad, Ts


def get_loss(data, params, indices=None, sterr=True):
    """ get RMSE of dG predictions """
    data.update_energies(params, indices)
    data.get_predictions(indices)
    return data.get_rmse(indices, sterr)


def recalc_ensembles(data, params):
    """ get data with ensembles recalculated with given param set """
    data.recalc_ensembles(params)
    return data


class RiboNetTrainer(object):
    """
    CNNs for nearest neighbor parameters from kd measurements
    adds kd calculations to extend standard CNN object
    """

    def __init__(self, data, n_units, save=False, n_cores=1,
                 learning_rate=None, batch_norm=False, name='',
                 low_mem=False, optimizer='rmsprop', num_gpus=1,
                 testdata=None, **kwargs):
        """ initialize data and conv net """
        if learning_rate is None:
            if isinstance(data, e.MeltDataset):
                learning_rate = 1e-6
            else:
                learning_rate = 1e-4
        self.save = save
        self.model = n.CNN(n_units, save, learning_rate, optimizer, name,
                           low_mem=low_mem, **kwargs)
        self.name = self.model.name
        self.num_gpus = num_gpus
        self.data = data
        self.testdata = testdata
        self.learning_rate = learning_rate
        if optimizer not in ['descent', 'adam', 'adagrad', 'rmsprop']:
            raise ValueError('optimizer must be \'descent\', \'adagrad\', '
                             '\'rmsprop\', or \'adam\'')
        self.optimizer = optimizer
        self.low_mem = low_mem
        self.params = pp.NNParams('%s/resources/nupack/parameters/%s'
                                  % (s.BASE_DIR, s.paramfile))
        self.n_cores = n_cores
        self.cache = {}
        self.update_params()

    def _get_prior(self, motif):
        return self.params.get_energy(motif, True)[0:2]

    def get_weights(self, layer):
        return self.model.get_weights(layer)

    def train(self, epochs, batch_size, keepprob=1., alpha=0, sterr=True):
        """ train given number of epochs with given batch size """
        if batch_size > self.data.n:
            raise ValueError('cannot have batch size greater than data size')
        if self.optimizer not in ['descent', 'adam', 'adagrad', 'rmsprop']:
            raise ValueError('optimizer must be \'descent\', \'adagrad\', '
                             '\'rmsprop\', or \'adam\'')
        batch_size *= self.num_gpus

        print 'baseline rmse', self.data.get_baseline_rmse(sterr)

        for i in range(int(self.data.n / batch_size * epochs)):
            # pick data points and motifs for batch
            if self.low_mem:
                r, motifs, grad, Ts = run_in_child_process(
                    get_train_batch,
                    (self.data, self.params, batch_size, i, sterr))
            else:
                r, motifs, grad, Ts = get_train_batch(
                    self.data, self.params, batch_size, i, sterr)

            grad = np.expand_dims(grad, 1)
            dGs = self.model.test(motifs)
            
            # dervatives wrt dG37 and dH
            relativeT = (Ts + e.CtoK) / (37 + e.CtoK)
            grad = np.tile(grad, 2) * np.hstack((relativeT, 1 - relativeT))

            # add regularization
            if alpha:
                prior = np.array([self._get_prior(motif) for motif in motifs])
                grad += alpha * (dGs - prior)

            # train batch of motifs
            self.model.train_batch(motifs, dGs - grad, keepprob)

            # store new motifs in params object and update energies
            self.params.add_motifs(motifs, dGs)
            self.update_params()

            if (i+1) % (self.data.n / batch_size) == 0:
                loss = self.data.get_loss(sterr=sterr)
                print 'step %d: rmse %g' % (i, loss)
                print '\tupdate ratio: %.1f' % np.log10(np.abs((
                    grad * self.learning_rate) / dGs)).mean()
                if testdata:
                    loss = self.testdata.get_test_loss(sterr=sterr)
                    print '\ttest rmse: %g' % loss

                # update ensembles every 10 epochs
                if (i+1) % (5 * self.data.n / batch_size) == 0:
                    filename = ''.join(random.choice(string.ascii_lowercase)
                                       for _ in range(6))
                    self.params.to_file('%s/%s' % (s.TEMP_DIR, filename))
                    
                    self.data.recalc_ensembles(filename)
                    os.system('rm %s/%s.d*' % (s.TEMP_DIR, filename))

                    if self.save:
                        self.save_model('_%.3f' % loss.sum())
            sys.stdout.flush()

        print 'final: rmse %g' % get_loss(self.data, self.params, sterr=sterr)

    def update_dataset(self, data):
        """ change to new dataset """
        self.data = data
        self.update_params()

    def add_motifs(self):
        """ add all motifs to parameter object """
        motifsdict = self.data.get_motifs_arrays()
        for l, motifs in motifsdict.iteritems():
            dGs = self.model.test(motifs)
            self.params.add_motifs(motifs, dGs)

    def update_params(self, ls=None):
        """ update parameters to reflect current net """
        if ls is None:
            ls = self.params.get_lengths()
        for l in ls:
            dGs = self.model.test(self.params.X[l])
            self.params.update_energies(l, dGs)

    def add_all_motifs(self, ls=None):
        """ add all relevant motifs to params """
        if ls is None:
            ls = self.params.get_lengths()
        for l in ls:
            motifs = self.data.get_motif_array(l)
            dGs = self.model.test(motifs)
            self.params.add_motifs(motifs, dGs)

    def get_loss(self, indices=None, sterr=True):
        """ get rmse of kd predictions for given indices """
        self.data.update_energies(self.params, indices)
        self.data.get_predictions(indices)
        return self.data.get_rmse(indices, sterr)

    def get_test_loss(self, indices=None, sterr=True):
        """ get rmse of test kd predictions for given indices """
        if self.testdata is None:
          return np.nan
        self.testdata.update_energies(self.params, indices)
        self.testdata.get_predictions(indices)
        return self.testdata.get_rmse(indices, sterr)

    def get_r2(self, indices=None):
        """ get r2 of predictions """
        return self.data.get_r2(indices)

    def write_results_file(self, filename):
        """ write results to file """
        self.data.to_file(filename)

    def save_model(self, suffix=''):
        """ save variables to file """
        self.model.save_model(suffix)
        pickle.dump(self.params,
                    open('%s/%s.params' % (s.MODELS_DIR, self.model.name),
                         'w'))
        pickle.dump(self.data,
                    open('%s/%s.dat' % (s.MODELS_DIR, self.model.name), 'w'))
        self.data.copy = True

    def restore(self, filename):
        """ restore variables from file """
        try:
            self.model.restore(filename)
        except Exception as e:
            warnings.warn('unable to restore CNN from file, continuing using'
                          ' random initialization\n%s' % e)
        basename = filename.rsplit('-', 1)[0]
        self.name = self.model.name
        paramfile = '%s/%s.params' % (s.MODELS_DIR, self.name)
        if os.path.isfile(paramfile):
            print 'restoring params file %s' % paramfile
            self.params = pickle.load(open(paramfile))
        self.add_all_motifs()
        self.update_params()

class MeltTrainer(RiboNetTrainer):
    """
    CNNs for nearest neighbor parameters from melt measurements
    adds kd calculations to extend standard CNN object
    """

    def __init__(self, data, n_units, save=False, learning_rate=1e-4,
                 name='', optimizer='rmsprop', num_gpus=1, testdata=None,
                 **kwargs):
        """ initialize data and conv net """
        self.save = save
        self.model = n.CNN(n_units, save, learning_rate, optimizer, name,
                           num_gpus=num_gpus, **kwargs)
        self.name = self.model.name
        self.num_gpus = num_gpus
        self.data = data
        self.testdata = testdata
        self.learning_rate = learning_rate
        if optimizer not in ['descent', 'adam', 'adagrad', 'rmsprop']:
            raise ValueError('optimizer must be \'descent\', \'adagrad\', '
                             '\'rmsprop\', or \'adam\'')
        self.optimizer = optimizer
        self.params = pp.NNParams('%s/resources/nupack/parameters/%s'
                                  % (s.BASE_DIR, s.paramfile))
        self.cache = {}
        self.add_all_motifs()
        self.update_params()

    def train(self, epochs, batch_size, keepprob=1., alpha=0, sterr=True, test=None):
        """ train given number of epochs with given batch size """
        if batch_size > self.data.n:
            raise ValueError('cannot have batch size greater than data size')
        if self.optimizer not in ['descent', 'adam', 'adagrad', 'rmsprop']:
            raise ValueError('optimizer must be \'descent\', \'adagrad\', '
                             '\'rmsprop\', or \'adam\'')

        batch_size *= self.num_gpus
        batch_size = min(batch_size, self.data.n)
        self.data.update_energies(self.params)
        self.data.get_predictions()
        print 'baseline rmse', self.data.get_baseline_rmse(sterr)
        epochlen = self.data.n / batch_size

        losses = []
        for i in range(epochlen * epochs):
            # get batch
            ls = {}
            while len(ls) == 0:
                r = np.random.choice(self.data.n, batch_size, replace=False)
                ls = self.data.get_motif_lengths(r)
            l = np.random.choice(ls.keys(), 1, p=ls.values())[0]

            # get proper motif shape
            motifs = self.data.get_motif_array_unique(l, r)

            # get gradients
            self.data.update_energies(self.params, r)
            grad = self.data.get_grad_losses(motifs, r, sterr)
            
            dGs = self.model.test(motifs)

            # add regularization
            if alpha:
                prior = np.array([self._get_prior(motif) for motif in motifs])
                grad += alpha * (dGs - prior)

            # train batch of motifs
            self.model.train_batch(motifs, dGs - grad, keepprob)

            # store new motifs in params object and update energies
            self.params.add_motifs(motifs, dGs)
            self.update_params()

            # get current loss every epoch
            if (i+1) % epochlen == 0:
                loss = self.get_loss(r, sterr)
                print 'epoch %d (step %d): rmse %s' % ((i+1)/(self.data.n/batch_size),
                    i+1, loss)
                if self.testdata:
                    loss = np.hstack([loss, self.get_test_loss(sterr=sterr)])
                    print '\ttest rmse %s' % loss[-2:]
                losses.append(loss)
                print '\tupdate ratio: %.1f' % np.log10(np.abs((
                    grad * self.learning_rate) / dGs)).mean()
                if (i+1) % (5 * epochlen) == 0 and self.save:
                    self.save_model('_%.3f' % loss[-2:].sum())
            sys.stdout.flush()

        print 'final: rmse %s' % get_loss(self.data, self.params, sterr=sterr)
        return np.array(losses)

    def restore(self, filename):
        """ restore variables from file """
        try:
            n.CNN.restore(self.model, filename)
        except Exception as e:
            warnings.warn('unable to restore CNN from file, continuing using'
                          ' random initialization\n%s' % e)
        basename = filename.rsplit('-', 1)[0]
        self.name = self.model.name
        print self.name
        paramfile = '%s/%s.params' % (s.MODELS_DIR, self.name)
        if os.path.isfile(paramfile):
            print 'restoring params file %s' % paramfile
            self.params = pickle.load(open(paramfile))
        self.add_all_motifs()
        self.update_params()
        print 'epoch 0 (step 0): rmse %s' % self.get_loss(sterr=False)
