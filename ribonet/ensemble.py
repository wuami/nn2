import settings as s
import numpy as np
import pandas as pd
import multiprocessing as mp
import random
import string
import re
import subprocess as sp
import os
import sys
import functools as ft
from scipy import optimize
from sklearn import linear_model
import params as pp
import pickle
import copy
import traceback

kd_ms2 = 1e-8
rt = 1.987e-3*310
beta = 1/rt
fmn_pairs = [10, 9, 8, -1, -1, -1, -1, -1, -1, 2, 1, 0]
kd_fmn = 10e-6
CtoK = 273.15


def rand_string(n=8):
    """ random string of lowercase characters of length n """
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))


def predict_kd(dG0, dGMS2, dGFMN, dGFMNMS2, FMN):
    """
    get MS2 kd prediction based on dG values and FMN concentration
    """
    num = (np.exp(-dGFMN*beta) + np.exp(-dGFMNMS2*beta))*(1+FMN) + \
        np.exp(-dGMS2*beta) + np.exp(-dG0*beta)
    denom = np.exp(-dGMS2*beta) + np.exp(-dGFMNMS2*beta)*(1+FMN)
    if np.isnan(num/denom) or np.isinf(num/denom):
        return np.nan
        print num, denom
        print dG0, dGMS2, dGFMN, dGFMNMS2
        raise ValueError('input results in undefined prediction')
    return kd_ms2 * num / denom


def get_ensemble(sequence, params=None, pickle=False, T=37):
    """ get ensemble object or pickle path """
    if params is None:
        params = s.molecule
    try:
        e = Ensemble(sequence, T=T, params=params)
        if pickle:
            e = get_pickle(e)
        return e
    except Exception as err:
        print 'error in ensemble calculation: %s' % err
        return None
  
def get_conf_from_series(series):
    """ get conformation obejct """
    return Conformation(series['sequence'], np.array([series['dG'], series['dH']]),
                        series['secstruct'], series['meltstruct'])

def get_pickle(obj):
    """ pickle and returning path """
    if not isinstance(obj, str):
        path = '%s/%s.p' % (s.TEMP_DIR, rand_string())
        with open(path, 'w') as f:
            pickle.dump(obj, f)
        return path
    return obj


def unpickle(obj):
    """ unpickle object """
    if not isinstance(obj, str) or not os.path.isfile(obj):
        return obj
    with open(obj) as f:
        return pickle.load(f)


def get_dG(dG37, dH, T):
    """ get dG at given temperature """
    return dH + (T + CtoK) * (dG37 - dH) / (37 + CtoK)


class Dataset(object):
    """
    stores all information related to an experimental dataset
    """

    def __init__(self, filenames, fields, pickle=True,
                 temp=False):
        """ intialize dataset with data from given filenames """
        self.pickle = pickle
        self.copy = False

        self.data = self.read_data(filenames, fields)
        self.set_molecule()

        # compute ensembles in parallel
        if s.molecule == 'dna':
            self.calc_ensembles('dna1998', temp=temp)
        else:
            self.calc_ensembles(temp=temp)

        # finish initialization
        if self.pickle:
            self.data['ensemble_obj'] = None
        else:
            self.data['ensemble_obj'] = self.data['ensemble']

    def read_data(self, filenames, fields):
        """
        read in specified fields from given filenames
        """
        data = pd.DataFrame()
        for filename in filenames:
            new_data = pd.read_csv(filename, sep='\t')
            if 'NumberOfClusters' in new_data.columns:
                new_data = new_data[new_data.NumberOfClusters > 20]
            try:
                new_data = new_data[fields]
            except KeyError as e:
                print '%s not read' % filename
                print e
                continue
            data = data.append(new_data)
        data = data.reset_index(drop=True)
        return data

    def set_molecule(self):
        """
        set molecule type
        """
        molecule = None
        if any(self.data.sequence.str.contains('U')):
            molecule = 'rna'
        if any(self.data.sequence.str.contains('T')):
            if molecule == 'rna':
                raise ValueError('input file contains both RNA and DNA '
                                 'sequences')
            molecule = 'dna'
        s.set_molecule(molecule)

    def check_bps(self):
        """ make sure all base pairs are valid """
        for i, row in self.data.iterrows():
            seq = row['sequence']
            ss = row['secstruct']
            openpairs = []
            for j in range(len(seq)):
                if ss[j] == '(':
                    openpairs.append(j)
                elif ss[j] == ')':
                    partner = openpairs.pop()
                    bp = seq[partner] + seq[j]
                    if bp not in s.bps:
                        raise ValueError('invalid base pair %s at row %d' %
                                         (bp, i))

    def get_data_frame(self, fields=None):
        """ returns data frame """
        if fields is None:
            fields = self.data.columns
        return self.data[fields]

    def to_file(self, filename, fields=None):
        """ write data to tab delimited file """
        if fields is None:
            fields = self.data.columns
        self.data[fields].to_csv(filename, sep='\t', index=False)

    def adjust_motif_energies(self, motifs, ddGs):
        """ adjust several motif energies in all ensembles in dataset """
        if motifs.shape[0] != ddG.shape[0]:
            raise ValueError('nonmatching shapes')
        for i in range(motifs.shape[0]):
            self.adjust_motif_energy(motifs[i, :, :], ddG[i])

    def adjust_motif_energy(self, motif, ddG):
        """ adjust energy for a single motif in all ensembles in dataset """
        try:
            self.data.ensemble_obj.apply(lambda e: e.adjust_motif_energy(motif,
                                                                         ddG))
        except AttributeError as e:
            print e
            raise ValueError('must unpickle ensemble %d before use' % i)


class MeltDataset(Dataset):
    """
    represents a set of experimental melting measurements
    """

    def __init__(self, filenames, pickle=True):
        """
        read in melt measurements from given filenames and calculate ensembles
        and motifs in given sequences
        """
        fields = ['fluor', 'fluor_sterr', 'sequence', 'T']
        Dataset.__init__(self, filenames, fields, pickle=pickle,
                         temp=True)
        self.n = self.data.shape[0]
        self.get_predictions()

    def to_file(self, filename):
        """ write data to tab delimited file """
        Dataset.to_file(self, filename, ['fluor', 'fluor_predicted',
                                         'fluor_sterr', 'T'])

    def calc_ensembles(self, params=None, temp=False):
        """ get all ensembles in parallel """
        if params is None:
            params = s.paramfile
        if not (isinstance(params, pp.NNParams) or
                os.path.isfile('%s.dG' % params) or
                os.path.isfile('%s/%s.dG' % (s.TEMP_DIR, params)) or
                params in ['rna1995', 'dna1998']):
            raise ValueError('%s is not a valid parameter file' % params)

        p = mp.Pool()
        # store results temporarily in global dictionary so that processes
        # don't block and memory clears
        done = mp.Manager().dict()

        def callback(value, i):
            """ callback function to keep track of finished processes """
            done[i] = value

        # get ensembles asynchronously
        results = []
        for i, row in self.data.iterrows():
            if temp:
                r = p.apply_async(get_ensemble, (row['sequence'], params,
                                                 self.pickle, row['T']),
                                  callback=ft.partial(callback, i=i))
            else:
                r = p.apply_async(get_ensemble, (row['sequence'],
                                                 params, self.pickle),
                                  callback=ft.partial(callback, i=i))
            results.append(r)

        # make sure all processes have finished
        for r in results:
            r.wait()
        p.close()
        p.join()

        # move results to data table
        for i, ensemble in done.items():
            if ensemble is None:
                continue
            self.data.ix[i, 'ensemble'] = ensemble

        assert sum(self.data.ensemble.apply(lambda x: x == np.nan)) == 0

    def _clean_pickle(self, path):
        """ remove pickle if not still being used """
        if path not in self.data.ensemble.values \
           and os.path.isfile(path):
            os.remove(path)

    def copy_ensembles(self):
        """ copy all ensembles and preserve old files """
        new_paths = {}
        for i, path in enumerate(self.data['ensemble']):
            if isinstance(path, str):
                if path in new_paths:
                    self.data.ix[i, 'ensemble'] = new_paths[path]
                else:
                    new_path = '%s/%s.p' % (s.TEMP_DIR, rand_string())
                    if sp.call(['cp', path, new_path]):
                        raise ValueError('ensemble file %s not found' %
                                         path)
                    new_paths[path] = new_path
                    self.data.ix[i, 'ensemble'] = new_path

    def recalc_ensembles(self, params):
        """ recalculate ensembles using specified parameter file """
        # get list of ensembles that are being overwritten
        paths = self.data['ensemble'].copy()

        # get new ensembles in parallel
        self.calc_ensembles(params)

        # delete files that are no longer needed
        if self.pickle:
            paths.apply(self._clean_pickle)
        self.n = self.data.shape[0]

    def update_energies(self, params, indices=None):
        """ update energies using specified parameter file """
        if indices is None:
            indices = range(self.n)
        for i in indices:
            T = self.data.ix[i, 'T']
            if self.pickle and self.data.ix[i, 'ensemble_obj'] is None:
                with open(self.data.ix[i, 'ensemble']) as f:
                    e = pickle.load(f)
                e.update_energies(params, T)
                with open(self.data.ix[i, 'ensemble'], 'w') as f:
                    pickle.dump(e, f)
            else:
                self.data.ix[i, 'ensemble_obj'].update_energies(params, T)

    def unpickle_ensembles(self, indices):
        """ unpickle given indices """
        if self.pickle:
            p = mp.Pool()
            self.data.ix[indices, 'ensemble_obj'] = p.map(
                unpickle, self.data.ix[indices, 'ensemble'])
            p.close()

    def pickle_ensembles(self, indices):
        """ pickle given indices """
        if self.pickle:
            self.data.ix[indices, 'ensemble'].apply(self._clean_pickle)
            p = mp.Pool()
            self.data.ix[indices, 'ensemble'] = p.map(
                get_pickle, self.data.ix[indices, 'ensemble_obj'])
            p.close()
            self.clear_ensembles(indices)

    def clear_ensembles(self, indices):
        """ clear ensembles for given indices from memory """
        if self.pickle:
            self.data.ix[indices, 'ensemble_obj'] = None

    def get_motif_lengths(self, indices=None):
        """
        get all possible motif lengths and proportion of total occurrences
        for given rows in data
        """
        if indices is None:
            indices = range(self.n)
        ls = {}
        for i in indices:
            try:
                m = self.data.ix[i, 'ensemble_obj'].motifs_arrays
                for l in m.keys():
                    if l not in ls:
                        ls[l] = 0
                    ls[l] += m[l].shape[0]
            except AttributeError as e:
                print e
                raise ValueError('must unpickle ensemble %d before use' % i)
        tot = float(sum(ls.values()))
        return {k: v/tot for k, v in ls.iteritems()}

    def get_motif_array(self, l, indices=None):
        """ get combined motif array for length l for given rows """
        if indices is None:
            indices = range(self.n)
        return np.vstack([self._get_motif_array_single(i, l) for i in indices])

    def get_temps(self, indices=None):
        if indices is None:
            return self.data['T'].values
        else:
            return self.data.ix[indices, 'T'].values

    def get_temps_motifs(self, indices, l):
        Ts = []
        for i in indices:
            T = self.data.ix[i, 'T']
            Ts.extend([T] * self._get_motif_array_single(i, l).shape[0])
        return np.array(Ts)

    def _get_motif_array_single(self, i, l):
        """ get motif array for data point i """
        try:
            if l not in self.data.ix[i, 'ensemble_obj'].motifs_arrays:
                return np.zeros((0, l, 8))
        except AttributeError as e:
            print e
            raise ValueError('must unpickle ensemble %d before use' % i)
        return self.data.ix[i, 'ensemble_obj'].motifs_arrays[l]

    def get_predictions(self, indices=None):
        """
        get fluorescence predictions for all data
        """
        if indices is None:
            self.data['fluor_predicted'] = self.data.apply(
                self.get_prediction, axis=1)
        else:
            self.data.ix[indices, 'fluor_predicted'] = self.data.ix[
                indices].apply(self.get_prediction, axis=1)

    def get_prediction(self, row):
        """
        get kd prediction for one row of data frame
        """
        if self.pickle and row.ensemble_obj is None:
            with open(row.ensemble) as f:
                e = pickle.load(f)
        else:
            e = row.ensemble_obj
        return e.get_predicted_melted(row['T'])

    def get_rmse(self, indices=None, sterr=True):
        """ get rmse of dGs """
        if indices is None:
            indices = range(self.n)
        if not np.any(np.isfinite(self.data.ix[indices, 'fluor_predicted'])):
            raise ValueError('fluorescence predictions are not finite')
        indices = [i for i in indices
                   if np.isfinite(self.data.ix[i, 'fluor_predicted'])]
        err = self.data.ix[indices, 'fluor'] - \
            self.data.ix[indices, 'fluor_predicted']
        if sterr:
            err /= self.data.ix[indices, 'fluor_sterr']
        return np.sqrt(np.mean(np.square(err)))

    def get_r2(self, indices=None):
        """ get r2 of dGs """
        if indices is None:
            indices = range(self.n)
        return np.square(np.corrcoef(self.data.fluor[indices],
                                     self.data.fluor_predicted[indices])[0, 1])

    def get_baseline_rmse(self, sterr=True):
        """ get baseline rmse, aka standard deviation of values """
        u = self.data['fluor'].mean()
        err = self.data['fluor'] - u
        if sterr:
            err /= self.data['fluor_sterr'].values
        return np.sqrt(np.mean(np.square(err)))

    def get_grad_losses(self, motifs, indices=None, sterr=True):
        """
        get gradient of loss for rows in i for multiple motifs
        loss function is weighted sum of squares
        """
        if indices is None:
            indices = range(self.n)
        losses = np.zeros(motifs.shape[0])
        for i in range(motifs.shape[0]):
            losses[i] = self.get_grad_loss(motifs[i, :, :], indices, sterr)
        return losses

    def get_grad_loss(self, motif, indices=None, sterr=True):
        """
        get gradient of loss for rows in i
        loss function is weighted sum of squares of fluorescences
        """
        if indices is None:
            indices = range(self.n)
        sum_ = 0
        denom = 0
        for i in indices:
            row = self.data.iloc[[i]].squeeze()
            try:
                grad = row.ensemble_obj.get_grad_melt(motif, row['T'])
            except AttributeError as e:
                print e
                raise ValueError('must unpickle ensemble %d before use' % i)
            if not np.isfinite(row.fluor_predicted):
                print 'predicted fluorescence for row %d is not finite' % i
                continue
            if not np.isfinite(grad):
                print 'gradient for row %d is not finite' % i
                continue
            if sterr:
                sum_ += grad / row.fluor_sterr**2 * (row.fluor_predicted -
                                                     row.fluor)
            else:
                sum_ += grad * (row.fluor_predicted - row.fluor)
            denom += 1
        return 2 * sum_ / denom

    def __del__(self):
        """ delete pickle files """
        if self.pickle and not self.copy:
            for e in self.data.ensemble:
                if os.path.isfile(e):
                    os.system('rm %s' % e)


class dGDataset(Dataset):
    """
    represents a set of experimental dG measurements
    """

    def __init__(self, filename):
        """
        read in kd measurements from given filenames and calculate ensembles
        and motifs in given sequences
        """
        # read in sequence and kd data from given filenames
        fields = ['dG', 'dH', 'dG_sterr', 'dH_sterr', 'secstruct', 'meltstruct',
                  'sequence']
        self.data = pd.read_csv(filename, sep='\t')[fields]
        self.set_molecule()
        self.check_bps()
        self.n = self.data.shape[0]
        
        p = mp.Pool()
        self.data['motifs'] = p.map(get_conf_from_series,
                                    [row[1] for row in self.data.iterrows()])
        self.data['dG_predicted'] = np.nan
        self.data['dH_predicted'] = np.nan
        self.get_predictions()

    def update_energies(self, params, indices=None):
        if indices is None:
            indices = range(self.n)
        self.data.motifs[indices].apply(lambda m: m.update_dGdH(params))

    def to_file(self, filename):
        """ write data to tab delimited file """
        self.data.to_csv(filename, sep='\t', index=False)

    def get_predictions(self, indices=None):
        """
        get predictions for all data
        """
        if indices is None:
            indices = range(self.n)
        self.data.ix[indices, ['dG_predicted', 'dH_predicted']] = np.vstack(
            self.data.motifs[indices].apply(lambda m: m.dG))

    def get_prediction(self, row):
        """
        get kd prediction for one row of data frame
        """
        return pd.Series(row['motifs'].dG)

    def get_linear_fit(self, regularize=2, params=None):
        """
        get linear fit of motifs
        if params is provided, it is used to remove energies of base pair
        steps from the calculation
        """
        X, motifs = self.get_motif_matrix()
        motifs = [m for motif in motifs.values() for m in motif]
        ddG = np.zeros([X.shape[0], 2])
        if params is not None:
            bpstacks = np.array([True if m.shape[0] == 4 and m[:,4].sum() == 2
                                 else False for i, m in enumerate(motifs)])
            dG = self.data.dG
            dH = self.data.dH
            for i in np.where(bpstacks)[0]:
                energies = params.get_energy(motifs[i])
                ddG += np.outer(X[:,i], energies)
            X = X[:,~bpstacks]
            motifs = [m for i, m in enumerate(motifs) if not bpstacks[i]]
        if regularize == 0:
            fit_dG = optimize.lsq_linear(X, self.data.dG - ddG[:,0])
            fit_dH = optimize.lsq_linear(X, self.data.dH - ddG[:,1])
            fit_dG, fit_dH = fit_dG.x, fit_dH.x
        else:
            if regularize == 1:
                clf = linear_model.Lasso()
            elif regularize == 2:
                clf = linear_model.Ridge(0.1)
            else:
                print('invalid value for regularize')
            fit_dG = clf.fit(X, self.data.dG - ddG[:,0]).coef_
            fit_dH = clf.fit(X, self.data.dH - ddG[:,1]).coef_
        self.data['dG_predicted'] = X.dot(fit_dG) + ddG[:,0]
        self.data['dH_predicted'] = X.dot(fit_dH) + ddG[:,1]
        print 'linear rmse: %s' % self.get_rmse(sterr=False)
        return motifs, fit_dG, fit_dH

    def get_motif_lengths(self, indices=None, counts=False):
        """
        get all possible motif lengths and proportion of total occurrences
        for given rows in data
        """
        if indices is None:
            indices = range(self.n)
        ls = {}
        for i in indices:
            m = self.data.ix[i, 'motifs'].get_motif_lengths()
            for l in m.keys():
                if l not in ls:
                    ls[l] = 0
                ls[l] += m[l]
        if counts:
            return ls
        else: 
            tot = float(sum(ls.values()))
            return {k: v/tot for k, v in ls.iteritems()}
    
    def get_motif_array(self, l, indices=None):
        """ get combined motif array for length l for given rows """
        if indices is None:
            indices = range(self.n)
        return np.vstack([self.data.ix[i, 'motifs'].motifs[l]
                          if l in self.data.ix[i, 'motifs'].motifs
                          else np.zeros((0, l, 8)) for i in indices])

    def get_motif_array_unique(self, l, indices=None):
        """ get combined motif array for length l without repeats"""
        motifs = self.get_motif_array(l, indices)
        if not motifs.shape[0]:
          return np.zeros((0, l, 8))
        motifsflat = np.reshape(motifs, [motifs.shape[0], -1])
        uniqueflat = np.vstack({tuple(row) for row in motifsflat})
        return np.reshape(uniqueflat, [-1, motifs.shape[1], motifs.shape[2]])

    def get_motif_matrix(self):
        """ get matrix of motif counts for each sequence """
        motifs = {}
        for l in self.get_motif_lengths(counts=True):
            motifs[l] = self.get_motif_array_unique(l)
        counts = []
        for l in motifs:
            for m in motifs[l]:
                counts.append([c.count_motif(m) for c in self.data['motifs']])
        return np.stack(counts).T, motifs

    def get_baseline_rmse(self, sterr=True):
        """ get baseline rmse, aka standard deviation of values """
        us = self.data[['dG', 'dH']].mean(axis=0)
        err = (self.data[['dG', 'dH']] - us).values
        if sterr:
            err /= self.data[['dG_sterr', 'dH_sterr']].values
        return np.sqrt(np.mean(np.square(err), axis=0))

    def get_rmse(self, indices=None, sterr=True):
        """ get rmse of dGs """
        if indices is None:
            indices = range(self.n)
        self.get_predictions(indices)
        err = self.data.ix[indices, ['dG', 'dH']].values - \
            self.data.ix[indices, ['dG_predicted', 'dH_predicted']].values
        if sterr:
            err /= self.data.ix[indices, ['dG_sterr', 'dH_sterr']].values
        return np.sqrt(np.mean(np.square(err), axis=0))

    def get_r2(self, indices=None):
        """ get r2 of dGs """
        if indices is None:
            indices = range(self.n)
        self.get_predictions(indices)
        return np.square(np.array(
            [np.corrcoef(self.data.dG[indices],
                         self.data.dG_predicted[indices])[0, 1],
             np.corrcoef(self.data.dH[indices],
                         self.data.dH_predicted[indices])[0, 1]]))

    def get_grad_losses(self, motifs, indices=None, sterr=True):
        """
        get gradient of loss for rows in i for multiple motifs
        loss function is weighted sum of squares
        """
        if indices is None:
            indices = range(self.n)
        losses = np.zeros((motifs.shape[0], 2))
        for i in range(motifs.shape[0]):
            losses[i, :] = self.get_grad_loss(motifs[i, :, :], indices, sterr)
        return losses

    def get_grad_loss(self, motif, indices=None, sterr=True):
        """
        get gradient of loss for rows in i
        loss function is weighted sum of squares of dGs
        """
        if indices is None:
            indices = range(self.n)
        sum_ = np.zeros((2,))
        counts = np.array([m.count_motif(np.squeeze(motif)) for m in
                           self.data.motifs[indices]])
        grad = (self.data.ix[indices, ['dG_predicted', 'dH_predicted']].values -
            self.data.ix[indices, ['dG', 'dH']].values)
        if sterr:
            grad /= np.square(self.data.ix[indices, ['dG_sterr',
                                                     'dH_sterr']].values)
        grad *= np.stack((counts, counts), axis=1)
        return 2 * grad[counts != 0, :].sum(axis=0)


class Ensemble(object):
    """
    represents an ensemble of possible conformation for a sequence
    """

    def __init__(self, sequence, T=37., params=None, gap=3):
        """ initialize ensemble for sequence with given parameter set """
        if params is None:
            params = s.paramfile
        self.sequence = sequence
        self.T = T

        # parse out various formats for input param string
        self.clean = False
        if not isinstance(params, str):
            try:
                filename = rand_string()
                params.to_file('%s/%s' % (s.TEMP_DIR, filename))
                params = '%s/%s' % (s.TEMP_DIR, filename)
                self.clean = params
            except AttributeError:
                raise ValueError('params argument must be a string or param '
                                 'object')
        elif params in ['rna1995', 'dna1998']:
            params = '%s/resources/nupack/parameters/%s' % (s.BASE_DIR,
                                                                 params)
        elif os.path.isfile('%s.dG' % params):
            pass
        elif os.path.isfile('%s/%s.dG' % (s.TEMP_DIR, params)):
            params = '%s/%s' % (s.TEMP_DIR, params)
        else:
            raise ValueError('not a valid parameter file %s' % params)

        # get conformations within gap kcal/mol of mfe
        self.conformations = []
        self._get_ensemble(gap, params)
        self.n_conf = len(self.conformations)
        self.motifs, self.motif_counts = self.get_ensemble_motifs()
        self.motifs_arrays = self.get_motifs_arrays()

        # get energies, whether or not ms2/fmn are formed for each conformation
        self.dG = np.array([c.dG for c in self.conformations], dtype=float)
        self.ms2 = np.array([c.forms_ms2() for c in self.conformations],
                            dtype=bool)
        self.fmn = np.array([c.forms_fmn() for c in self.conformations],
                            dtype=bool)
        self.closing = np.array([c.forms_closing_pair() for c in
                                 self.conformations],
                                dtype=bool)

    def _get_ensemble(self, gap, params):
        """ run nupack subopt to get suboptimal structures """
        name = '%s/%s' % (s.TEMP_DIR, rand_string())
        with open('%s.in' % name, 'w') as f:
            f.write('%s\n' % self.sequence)
            f.write('%f\n' % gap)
        p = sp.Popen([os.path.join(s.RESOURCES_DIR, 'nupack', 'subopt'),
                     '-material', params, '-T', str(self.T), name],
                     stderr=sp.STDOUT)
        stdout, stderr = p.communicate()
        if p.returncode:
            print stdout, stderr
            print self.sequence
            os.system('rm %s*' % name)
            raise Exception('ensemble calculation errored out, try a lower '
                            'learning rate')
        with open('%s.subopt' % name, 'r') as f:
            line = f.readline()
            while line:
                if line.startswith('% %'):
                    f.readline()
                    c = Conformation(self.sequence, float(f.readline()),
                                     f.readline().strip())
                    self.conformations.append(c)
                    # limit to 200 conformations
                    if len(self.conformations) >= 200:
                        break
                    line = f.readline()
                    while not line.startswith('% %'):
                        line = f.readline()
                line = f.readline()
        os.system('rm %s*' % name)

    def get_ensemble_motifs(self):
        """
        get boolean array representing conformations containing each motif
        present in the ensemble
        """
        motif_counts = {}
        motifs = {}
        for i, c in enumerate(self.conformations):
            for l in c.motifs:
                if l not in motif_counts:
                    motif_counts[l] = 0
                motif_counts[l] += c.motifs[l].shape[0]
                for j in range(c.motifs[l].shape[0]):
                    motif = c.motifs[l][j, :, :]
                    hash_ = pp.get_np_hash(motif)
                    if hash_ not in motifs:
                        motifs[hash_] = EnsembleMotif(motif, self.n_conf)
                    motifs[hash_].add_conformation(i)
        return motifs, motif_counts

    def get_motifs_arrays(self):
        """ get all motifs as np array """
        motifs_arrays = {}
        for l in self.motif_counts.keys():
            motifs_arrays[l] = np.stack([motif.mat for motif in
                                         self.motifs.values() if
                                         motif.mat.shape[0] == l])
        return motifs_arrays

    def update_energies(self, params, T=37):
        """ update energies to align with parameters """
        self.dG = np.array([c.update_energy(params, T) for c in
                            self.conformations])

    def adjust_motif_energy(self, motif, ddG):
        """ adjust ensemble energies to account for change in motif energy """
        h = pp.get_np_hash(motif)
        if h in self.motifs:
            self.dG[self.motifs[h].conformations] += ddG

    def get_dg(self, condition):
        """ get dG for conformations given by condition """
        return -rt*np.log(self.get_pf(condition))

    def get_pf(self, condition=None, T=37):
        """ get partition function for conformations given by condition """
        rt_tmp = rt * (T + CtoK) / (37 + CtoK)
        if condition is None:
            condition = np.ones(self.dG.shape, dtype=bool)
        return np.sum(np.exp(-self.dG[condition]/rt_tmp))

    def get_predicted_melted(self, T):
        """ get predicted proportion with closing base pair melted """
        Z_up = self.get_pf(~self.closing, T=T)
        Z_all = self.get_pf(T=T)
        return Z_up/Z_all

    def get_grad_melt(self, motif, T=37):
        """ get partial derivative of proportion melted wrt motif energy """
        h = pp.get_np_hash(motif)
        if h not in self.motifs:
            return 0

        # calculate partition functions
        Z_up = self.get_pf(~self.closing, T=T)
        Z_all = self.get_pf(T=T)
        Z_motif = self.get_pf(self.motifs[h].conformations, T=T)
        Z_motif_up = self.get_pf(np.logical_and(self.motifs[h].conformations,
                                                ~self.closing), T=T)
        if np.isinf(Z_all) or Z_all == 0:
            raise ValueError('numeric overflow, try decreasing the learning '
                             'rate')

        return - beta * (37 + CtoK) / (T + CtoK) * (Z_motif_up / Z_all - \
                                                  Z_up * Z_motif / Z_all**2)

    def get_predicted_kd(self, fmn_conc):
        """ get predicted kd value """
        kd = predict_kd(self.get_dG0(), self.get_dGMS2(), self.get_dGFMN(),
                        self.get_dGFMNMS2(), fmn_conc)
        return kd

    def get_grad_kd(self, motif, fmn_conc):
        """ get partial derivative of kd wrt motif energy """
        h = pp.get_np_hash(motif)
        if h not in self.motifs:
            return 0

        # calculate partition functions
        Z_motif = self.get_pf(self.motifs[h].conformations, fmn_conc)
        Z_ms2 = self.get_pf(self.ms2, fmn_conc)
        Z_motif_ms2 = self.get_pf(np.logical_and(self.motifs[h].conformations,
                                                 self.ms2), fmn_conc)
        Z_all = self.get_pf(np.ones(self.n_conf, dtype=bool), fmn_conc)

        # handle edge cases
        if Z_ms2 == 0:
            return 0

        # calculate probabilities
        p_motif = Z_motif / Z_all
        p_motif_ms2 = Z_motif_ms2 / Z_ms2
        p_ms2 = Z_ms2 / Z_all
        return - beta * (p_motif - p_motif_ms2) / p_ms2 * kd_ms2

    def get_dG0(self):
        """ get dG for partition function with no MS2 and no FMN """
        return self.get_dg(np.logical_and(~self.ms2, ~self.fmn))

    def get_dGMS2(self):
        """ get dG for partition function with MS2 and no FMN """
        return self.get_dg(np.logical_and(self.ms2, ~self.fmn))

    def get_dGFMN(self):
        """ get dG for partition function with no MS2 and FMN """
        return self.get_dg(np.logical_and(~self.ms2, self.fmn))

    def get_dGFMNMS2(self):
        """ get dG for partition function with MS2 and FMN """
        return self.get_dg(np.logical_and(self.ms2, self.fmn))

    def __del__(self):
        """ remove parameter files """
        if self.clean:
            if os.path.isfile('%s.dG' % self.clean):
                os.remove('%s.dG' % self.clean)


class EnsembleMotif(object):
    """
    represents a motif that can occur in the secondary structure of a sequence
    in the context of an ensemble of conformations
    """

    def __init__(self, mat, n):
        """
        initalize motif with matrix representation and empty list of
        conformations containing that motif
        """
        self.mat = mat
        self.conformations = np.zeros(n, dtype=bool)

    def add_conformation(self, i):
        """ add conformation that contains the motif """
        self.conformations[i] = 1

    def merge(self, motif):
        """ merge conformations from another motif object """
        if not np.allclose(motif.mat, self.mat):
            raise ValueError('given motif object is for a different motif')
        self.conformations = np.logical_and(self.conformations,
                                            motif.conformations)


class Conformation(object):
    """
    represents a single possible conformation for a sequence
    """

    def __init__(self, sequence, dG, secstruct, meltstruct=None):
        """ initialize conformation """
        if len(sequence) != len(secstruct):
            raise ValueError('sequence and secondary structure must be the '
                             'same length')
        self.dG = dG
        self.sequence = sequence
        self.secstruct = secstruct
        self.motifs, self.alts, self.meltmotifs, self.meltalts = get_motifs(
            sequence, secstruct, meltstruct)
        self.ls = self.motifs.keys() + self.meltmotifs.keys()
        self.weights = {}
        for l in self.ls: 
            self.weights[l] = []
            if l in self.motifs:
                self.weights[l] += [1] * self.motifs[l].shape[0]
            if l in self.meltmotifs:
                self.weights[l] += [-1] * self.meltmotifs[l].shape[0]

    def update_dGdH(self, params):
        """ update thermodynamic parameters based on given parameter set """
        self.dG = np.zeros((2,))
        for l in self.motifs:
            for i in range(self.motifs[l].shape[0]):
                self.dG += params.get_energy(self.motifs[l][i, :, :])
        return self.dG

    def update_energy(self, params, T):
        """ update energy of conformation based on given parameter set """
        self.dG = self.get_energy(params, T)
        return self.dG

    def get_energy(self, params, T=37., print_=False, melt=False):
        """ get energy of conformation based on given parameter set """
        energies = {}
        for l in self.ls:
            energies[l] = []
            if l in self.motifs:
                for i in range(self.motifs[l].shape[0]):
                    c = params.get_energy(self.motifs[l][i, :, :])
                    energies[l].append(get_dG(c[0], c[1], T))
            if melt:
                if l in self.meltmotifs:
                    for i in range(self.meltmotifs[l].shape[0]):
                        c = params.get_energy(self.meltmotifs[l][i, :, :])
                        energies[l].append(get_dG(c[0], c[1], T))
        for choices in self.alts:
            e = []
            for choice in choices:
                e.append(sum([energies[l][j] for l, j in choice]))
            best = np.argmin(e)
            for i, choice in enumerate(choices):
                if i == best:
                    for l, j in choice:
                        self.weights[l][j] = 1
                else:
                    for l, j in choice:
                        self.weights[l][j] = 0
        if melt:
            for choices in self.meltalts:
                e = []
                for choice in choices:
                    e.append(sum([energies[l][j] for l, j in choice]))
                best = np.argmin(e)
                for i, choice in enumerate(choices):
                    if i == best:
                        for l, j in choice:
                            if l in self.motifs:
                                j += self.motifs[l].shape[0]
                            self.weights[l][j] = -1
                    else:
                        for l, j in choice:
                            if l in self.motifs:
                                j += self.motifs[l].shape[0]
                            self.weights[l][j] = 0
        total = 0.
        for l in self.motifs:
            for i in range(len(energies[l])):
                total += energies[l][i] * self.weights[l][i]
                if print_:
                    print energies[l][i], self.weights[l][i]
                    if i < self.motifs[l].shape[0]:
                        print self.motifs[l][i, :, :]
                    else:
                        print self.meltmotifs[l][i - self.motifs[l].shape[0], :, :]
        return total

    def get_motif_lengths(self):
        """ return number of motifs of each length as dictionary """
        return {l: motifs.shape[0] for l, motifs in self.motifs.iteritems()}

    def count_motif(self, mat):
        """ return number of occurences of motif """
        if mat.shape[0] not in self.motifs:
            return 0
        count = 0
        if mat.shape[0] not in self.motifs:
            return 0
        for motif in self.motifs[mat.shape[0]]:
            if np.array_equal(motif, mat):
                count += 1
        return count

    def has_motif(self, mat):
        """
        return boolean indicating whether conformation has given motif
        """
        for l in self.motifs:
            for motif in self.motifs[l]:
                if np.array_equal(motif, mat):
                    return True
        return False

    def forms_closing_pair(self):
        """
        return boolean indicating whether conformation forms closing base pair
        """
        if not hasattr(self, 'closing'):
            self.closing = True
            if self.secstruct[0] == '.' and self.secstruct[-1] == '.':
                self.closing = False
        return self.closing

    def forms_ms2(self):
        """
        return boolean indicating whether conformation forms ms2 haiprin
        """
        if not hasattr(self, 'ms2'):
            pos = [m.start() for m in re.finditer('ACAUGAGGAUCACCCAUGU',
                                                  self.sequence)]
            self.ms2 = False
            for i in pos:
                if self.secstruct[i:i+19] == '(((((.((....)))))))':
                    self.ms2 = True
                    break
        return self.ms2

    def forms_fmn(self):
        """
        return boolean indicating whether conformation forms fmn aptamer
        """
        if not hasattr(self, 'fmn'):
            pairs = get_pairs_from_secstruct(self.secstruct, self.sequence)
            pos1 = [m.start()-3 for m in re.finditer('AGGAUAU', self.sequence)]
            pos2 = [m.start()-2 for m in re.finditer('AGAAGG', self.sequence)]
            self.fmn = False
            for i in pos1:
                for j in pos2:
                    if i >= 0 and j >= 0:
                        # if longer half comes first
                        if self.secstruct[i:i+12] == '(((......(((' and \
                           self.secstruct[j:j+11] == '))).....)))':
                            self.fmn = True
                            for k in range(12):
                                if fmn_pairs[k] == -1:
                                    if pairs[i+k] != -1:
                                        self.fmn = False
                                elif pairs[i+k] != j+fmn_pairs[k]:
                                    self.fmn = False
                            if self.fmn:
                                return self.fmn
                        # if shorter half comes first
                        elif (self.secstruct[i:i+12] == ')))......)))' and
                              self.secstruct[j:j+11] == '(((.....((('):
                            self.fmn = True
                            for k in range(12):
                                if fmn_pairs[k] == -1:
                                    if pairs[i+k] != -1:
                                        self.fmn = False
                                elif pairs[i+k] != j+fmn_pairs[k]:
                                    self.fmn = False
                            if self.fmn:
                                return self.fmn
        return self.fmn


def get_pairs_from_secstruct(secstruct, seq):
    """
    return array with pairing partner for each base or negative if unpaired
    -5 for 5' dangle, -3 for 3' dangle, -1 otherwise
    """
    open_pairs = []
    pairs = [-1] * len(seq)
    if secstruct is None:
        return pairs
    for i in range(len(secstruct)):
        if secstruct[i] == '(':
            open_pairs.append(i)
        elif secstruct[i] == ')':
            try:
                j = open_pairs.pop()
            except:
                raise ValueError('improper pairing in %s' % secstruct)
            #if seq[i] + seq[j] not in s.bps:
            #    print seq
            #    print secstruct
            #    raise ValueError('%s and %s cannot pair' % (seq[i], seq[j]))
            pairs[i] = j
            pairs[j] = i

    # set dangles
    i = 0
    while i < len(secstruct) and pairs[i] == -1:
        pairs[i] = -5
        i += 1
    i = len(secstruct)-1
    while i >= 0 and pairs[i] == -1:
        pairs[i] = -3
        i -= 1
    return pairs


def get_motifs(sequence, secstruct, meltstruct=None):
    """
    get all sequence motifs in given structure
    return matrix for each motif where columns are
        A, C, G, U, (, ), 5' dangle, 3' dangle
    """
    if len(sequence) != len(secstruct):
        raise ValueError('sequence and secondary structure must be the same '
                         'length')
    pairs = get_pairs_from_secstruct(secstruct, sequence)
    meltpairs = get_pairs_from_secstruct(meltstruct, sequence)

    motifs = {}
    add_terminal_AUGU(motifs, sequence, pairs)
    bases = get_motifs_recursive(motifs, sequence, pairs, 0, len(secstruct)-1,
                                 meltpairs)
    alts = get_motifs_open(motifs, sequence, pairs, bases)
    meltmotifs = {}
    meltalts = get_motifs_open(meltmotifs, sequence, meltpairs)
    return motifs, alts, meltmotifs, meltalts


def get_open_bases(pairs):
    """ get bases in open external region of structure """
    i = 0
    curr = {}  # keep track of bases curr since last motif added
    while i < len(pairs):
        if pairs[i] < 0:
            curr[i] = pairs[i]
            i += 1
        else:
            curr[i] = pairs[i]
            i = pairs[i] + 1
    return curr


def get_motifs_open(motifs, sequence, pairs, bases=None):
    """ get motifs from open part of structure """
    if bases is None:
        bases = get_open_bases(pairs)
    alts = []
    if len(bases) > 2:
        base = min(bases.keys())
        curr = {}
        # keep track of motifs added in case they will be involved in coaxial
        # stacking
        motif_hist = {}
        # keep track of cases with multiple alternatives for energy calculation
        while base in bases:
            if bases[base] > 0:
                partner = bases[base]
                bp_stored = False
                # 5' dangles
                if len(curr) > 0:
                    curr[base] = partner
                    curr[partner] = base
                    motif_hist[(base, 5)] = add_motif(motifs, sequence, curr)
                    bp_stored = True
                # 3' dangles
                if partner+1 in bases and bases[partner+1] < 0:
                    motif_hist[(base, 3)] = add_motif(
                        motifs, sequence, {base: partner, partner: base,
                                           partner+1: -3})
                    bp_stored = True
                # base pairs for coaxial stacking
                if not bp_stored:
                    motif_hist[(base,)] = None
                # check for coaxial stacking
                prev_pairs = list(set([pair[0] for pair in motif_hist
                                       if pair[0] != base]))
                if len(prev_pairs):
                    assert len(prev_pairs) == 1
                    i = prev_pairs[0]
                    # flush coaxial stacking
                    if base - pairs[i] == 1:
                        options = [[x for x in motif_hist.values()
                                    if x is not None]]
                        options.append([add_motif(motifs, sequence,
                                                  {i: pairs[i], pairs[i]: i,
                                                   base: partner,
                                                   partner: base})])
                        alts.append(options)
                    # mismatch mediated coaxial stacking
                    elif base - pairs[i] == 2:
                        options = [[x for x in motif_hist.values()
                                    if x is not None]]
                        # mismatch with 5' dangle
                        if (i, 5) in motif_hist:
                            coaxial_stack = [
                                add_motif(motifs, sequence,
                                          {i: pairs[i], pairs[i]: i, i-1: -5,
                                           pairs[i]+1: -3}),
                                add_motif(motifs, sequence,
                                          {base: partner, partner: base,
                                           i-1: -3, pairs[i]+1: -5})]
                            options.append(coaxial_stack)
                        # mismatch with 3' dangle
                        if partner+1 in bases and bases[partner+1] < 0:
                            coaxial_stack = [
                                add_motif(motifs, sequence,
                                          {i: pairs[i], pairs[i]: i,
                                           partner+1: -5, pairs[i]+1: -3}),
                                add_motif(motifs, sequence,
                                          {base: partner, partner: base,
                                           partner+1: -3, pairs[i]+1: -5})]
                            options.append(coaxial_stack)
                        if len(options) > 0:
                            alts.append(options)
                    else:
                        break
                    motif_hist = {k: v for k, v in motif_hist.iteritems()
                                  if k == base}
                if partner+1 in bases and bases[partner+1] < 0:
                    curr = {partner+1: -3}
                else:
                    curr = {}
                base = partner+1
            else:
                curr = {base: -5}
                base += 1
    return alts


def add_terminal_AUGU(motifs, sequence, pairs):
    """ add terminal AU and GU pairs to motif dict"""
    for i in range(len(sequence)):
        if if_terminal_AUGU(i, sequence, pairs):
            add_motif(motifs, sequence, {i: pairs[i], pairs[i]: i})


def if_terminal_AUGU(pos, sequence, pairs):
    """ check if position forms opening of terminal AU or GU pair """
    if pos > pairs[pos]:
        return False
    pair = sequence[pos] + sequence[pairs[pos]]
    if pair not in ['AU', 'UA', 'GU', 'UG']:
        return False

    # check for normal helix start and end pairs
    helix_start = pos == 0 or pairs[pos] == len(sequence)-1 or \
        pairs[pos-1] != pairs[pos]+1
    helix_end = pairs[pos+1] != pairs[pos]-1
    hairpin = pairs[pos+1:pairs[pos]] == [-1]*(pairs[pos]-pos-1)
    if helix_start:
        if not if_no_bonus(pos, pairs, -1):
            return True
    if helix_end:
        if not hairpin and not if_no_bonus(pos, pairs, 1):
            return True
        if hairpin and pairs[pos]-pos-1 == 3:
            return True
    return False


def if_no_bonus(pos, pairs, direction):
    """
    check if position forms part of no AT bonus features:
        1x1 1x2 or 2x2 internal loop (already accounted for in bonus)
        1 nt bulge (treat as stack)
    """
    pos1 = get_next_paired(pos, pairs, direction)
    pos2 = get_next_paired(pairs[pos], pairs, -direction)
    len1 = (pos1-pos)*direction - 1
    len2 = (pairs[pos]-pos2)*direction - 1

    # single bugles
    if (len1, len2) in [(0, 1), (1, 0)]:
        return True

    # internal loops
    if len1 > 0 and len2 > 0:
        return True
    return False


def get_next_paired(pos, pairs, direction):
    """ get position of next paired base in given direction """
    curr = pos
    while pairs[curr] == -1 or curr == pos:
        curr += direction
        if curr < 0 or curr > len(pairs) - 1:
            return direction * -999
    return curr


def get_motifs_recursive(motifs, sequence, pairs, start, end, meltpairs=None):
    """
    get all sequence motifs between start and end indices
    """
    i = start
    curr = {}  # keep track of bases curr since last motif added
    while i <= end:
        if pairs[i] < 0:
            curr[i] = pairs[i]
            i += 1
        else:
            curr[i] = pairs[i]
            if i < pairs[i]:
                if i != start:
                    if meltpairs is None or meltpairs[i] != pairs[i]:
                        get_motifs_recursive(motifs, sequence, pairs, i,
                                             pairs[i], meltpairs)
                    curr[pairs[i]] = i
                    i = pairs[i] + 1
                else:
                    i += 1
            elif i > pairs[i]:
                if pairs[i] in curr:
                    add_motif(motifs, sequence, curr)
                    curr = {pairs[i]: i, i: pairs[i]}
                i += 1
    return curr


def add_motif(motifs, sequence, bases):
    """
    add motif described by dict of bases to motif dict
    """
    n = len(bases)
    X = np.zeros((1, n, 8))

    # get start of motif
    if -5 in bases.values():
        base = [k for k, v in bases.iteritems() if v == -5][0]
    else:
        base = min(bases.keys())
    first_pair = base
    while first_pair in bases and bases[first_pair] < 0:
        first_pair += 1

    # loop through motif
    for i in range(n):
        X[0, i, s.bases[sequence[base]]] = 1
        if bases[base] < 0:
            if bases[base] == -5:
                X[0, i, 6] = 1
            elif bases[base] == -3:
                X[0, i, 7] = 1
        else:
            # special case for first pair
            if base == first_pair and base + 1 in bases:
                X[0, i, 5] = 1
            elif bases[base] == first_pair and bases[base] + 1 in bases:
                X[0, i, 4] = 1
            # otherwise first base is (
            elif bases[base] > base:
                X[0, i, 4] = 1
            elif bases[base] < base:
                X[0, i, 5] = 1
            else:
                assert False, 'improper form of motif'
        if base + 1 in bases:
            base += 1
        elif base in bases and bases[base] in bases:
            base = bases[base]
        else:
            base = min([b for b in bases.keys() if b != base])

    # add to motifs dict
    if n not in motifs:
        motifs[n] = X
    else:
        motifs[n] = np.append(motifs[n], X, 0)

    return n, motifs[n].shape[0] - 1
