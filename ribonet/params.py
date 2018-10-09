import settings as s
import numpy as np
from scipy import misc
import re
import sys
from os import path

def get_np_hash(array):
    """ get hash for a np array """
    array = array.astype(bool)
    array.flags.writeable = False
    return hash(array.data)


class NNParams(object):
    """
    represents a set of nearest neighbor model parameter
    """

    def __init__(self, filename, molecule=None, split=False):
        """
        parse nn data from file and create train and test sets
        """
        self.name = path.basename(filename)
        if molecule is None:
            if self.name.startswith('rna'):
                self.molecule = 'rna'
            elif self.name.startswith('dna'):
                self.molecule = 'dna'
            else:
                raise ValueError('filename must start with "rna" or "dna" or '
                                 'molecule must be specified')
        elif molecule not in ['rna', 'dna']:
            raise ValueError('molecule must be either "rna" or "dna"')

        self.X, self.y = self.parse_nupack_params(filename)
        self.n = sum([v.shape[0] for v in self.X.values()])
        self.hash_table = self.get_hash_table()
        if split:
            self.split_train_test()
            self.get_train_sizes()

    def split_train_test(self):
        """
        split input X and y into a training and test dataset
        """
        self.X_train = {}
        self.y_train = {}
        self.X_test = {}
        self.y_test = {}
        for l in self.X:
            n = self.X[l].shape[0]
            print '%d points of length %d' % (n, l)
            indices = np.random.permutation(n)
            split = int(round(0.8*n))
            train_idx, test_idx = indices[:split], indices[split:]
            self.X_train[l] = self.X[l][train_idx, :, :]
            self.y_train[l] = self.y[l][train_idx, :]
            self.X_test[l] = self.X[l][test_idx, :, :]
            self.y_test[l] = self.y[l][test_idx, :]
        return

    def get_train_sizes(self):
        """
        get proportion of training set with each length
        """
        self.p = {}
        try:
            for l in self.X:
                self.p[l] = self.X_train[l].shape[0]
        except AttributeError:
            raise AttributeError('no train/test split created')
        total = float(sum(self.p.values()))
        self.p = {k: v/total for k, v in self.p.items()}
        return

    def get_train_batch(self, n=100, l=None):
        """
        get training batch of size n of fixed, randomly selected size
        """
        try:
            if l is None:
                l = np.random.choice(self.p.keys(), p=self.p.values())
            n = min(self.X_train[l].shape[0], n)
            r = np.random.choice(self.X_train[l].shape[0], n, replace=False)
            X = np.reshape(self.X_train[l][r, :, :], (n, l, 8))
            y = np.reshape(self.y_train[l][r, :], (n, 2))
        except AttributeError:
            raise AttributeError('no train/test split created')
        return X, y

    def get_train(self, l=5):
        """ get train data """
        try:
            n = self.X_train[l].shape[0]
            X = np.reshape(self.X_train[l], (n, l, 8))
            y = np.reshape(self.y_train[l], (n, 2))
        except AttributeError:
            raise AttributeError('no train/test split created')
        return X, y

    def get_test(self, l=5):
        """ get test data """
        try:
            n = self.X_test[l].shape[0]
            X = np.reshape(self.X_test[l], (n, l, 8))
            y = np.reshape(self.y_test[l], (n, 2))
        except AttributeError:
            raise AttributeError('no train/test split created')
        return X, y

    def get_lengths(self):
        """ get all motif lengths represented """
        return self.hash_table.keys()

    def get_size(self):
        """ get number of total motifs stored """
        return sum([len(self.hash_table[l]) for l in self.hash_table.keys()])

    def get_hash_table(self):
        """
        get energies as hash table, where key is hashed motif matrix and value
        is list of the form [prior mean, current]
        """
        hash_table = {}
        for l in self.X:
            hash_table[l] = {}
            for i in range(self.X[l].shape[0]):
                hash_table[l][get_np_hash(self.X[l][i, :, :])] = np.tile(
                    self.y[l][i, :], 2)
        return hash_table

    def add_motifs(self, X, dGs):
        """ add motifs """
        assert dGs.shape[0] == X.shape[0], 'dGs shape incorrect'
        X = X.astype(bool)
        l = X.shape[1]
        if l not in self.X:
            self.X[l] = np.zeros((0, l, 8), dtype=bool)
            self.hash_table[l] = {}
        for i in range(dGs.shape[0]):
            h = get_np_hash(X[i, :, :])
            if h not in self.hash_table[l]:
                self.X[l] = np.append(self.X[l], X[None, i, :, :], 0)
                self.hash_table[l][h] = np.hstack(
                    (self.get_energy(X[i, :, :], prior=True)[-2:], dGs[i,:]))
            else:
                self.hash_table[l][h][-2:] = dGs[i,:]

    def update_energies(self, l, dGs):
        """ update energies for length l """
        assert dGs.shape[0] == self.X[l].shape[0], 'dGs shape incorrect'
        for i in range(dGs.shape[0]):
            self.update_energy(self.X[l][i, :, :], dGs[i,:])

    def update_energy(self, motif, dG):
        """ update energy for one motif """
        h = get_np_hash(motif)
        l = motif.shape[0]
        if h not in self.hash_table[l]:
            raise ValueError('motif not present in hash table')
        self.hash_table[l][h][-2:] = dG

    def get_energies(self, motifsdict, prior=False):
        """
        get energies for several motifs stored in a dictionary keyed by length
        """
        es = []
        for l, motifs in motifsdict.iteritems():
            for motif in motifs:
                es.append(self.get_energy(motif, prior=prior))        
        return np.stack(es)

    def get_energy(self, motif, prior=False):
        """
        get energy associated with a motif
        """
        # check for the motif in the dataset
        l = motif.shape[0]
        match = self.find_matching_motif(motif, prior=prior)
        if match is not None:
            return match
        # multiloops
        if np.sum(motif[:, 4]) > 2:
            p = np.sum(motif[:, 4])
            u = l - 2*p
            e = self.mloop_params[0] + p*self.mloop_params[1] + \
                u*self.mloop_params[2]
        elif motif[0, 5] == 1 and motif[-1, 4] == 1:
            # stem loops
            if np.all(motif[1:-1, 4:5] == 0):
                size = min(motif.shape[0]-3, 29)
                e = self.hairpins[size] + self.get_mismatch_bonus(motif)
            # bulges
            elif motif[1, 4] == 1 or motif[-2, 5] == 1:
                size = min(motif.shape[0]-5, 29)
                if size == 0:
                    pos = np.where(np.all(motif[:, 4:6] == 0, 1))
                    stack = self.find_matching_motif(np.delete(motif, pos, 0))
                    assert stack is not None, \
                       'error in bulge energy\n%s' % np.array_str(motif)
                    e = self.bulges[size] + stack
                else:
                    e = self.bulges[size]
            # interior loops
            elif np.sum(motif[:, 4]) == 2:
                _, m, n = get_iloop_lens(motif)
                e = self._get_iloop_energy(m, n) + \
                    self.get_mismatch_bonus(motif)
        # mismatch
        elif motif[0, 6] == 1 and motif[-1, 7] == 1:
            e = self._get_mismatch(motif, 1, 2, 0, 3)
        else:
            e = np.array([0,0])

        if prior:
            return np.hstack((e,e))
        else:
            return e

    def find_matching_motif(self, motif, bonus=True, prior=False):
        """ get energy of motif """
        l = motif.shape[0]
        if l in self.hash_table:
            for _ in range(l):
                h = get_np_hash(motif)
                if h in self.hash_table[l]:
                    if bonus:
                        try:
                            e = self.hash_table[l][h] + \
                                np.tile(self.get_mismatch_bonus(motif), 2)
                        except:
                            print self.hash_table[l][h]
                            print np.tile(self.get_mismatch_bonus(motif), 2)
                            sys.exit()
                    else:
                        e = self.hash_table[l][h]
                    if prior:
                        return e
                    else:
                        return e[-2:]
                motif = np.roll(motif, 1, axis=0)
        return None

    def get_mismatch_bonus(self, motif):
        l = motif.shape[0]
        pairs = np.sum(motif[:, 4])
        # hairpin loop
        if pairs == 1 and motif[0, 5] == 1 and motif[-1, 4] == 1:
            if l == 5:
                return np.array([0,0])
            return self._get_mismatch(motif, 0, -1, 1, -2, False)
        # internal loop
        if pairs == 2 and motif[0, 5] == 1 and motif[-1, 4] == 1 and l > 5:
            pos, m, n = get_iloop_lens(motif)
            if (m, n) not in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                bonus = []

                # first mismatch
                bonus.append(self._get_mismatch(motif, 0, -1, 1, -2))

                # second mismatch
                bonus.append(self._get_mismatch(motif, pos, pos+1, pos-1,
                                                pos+2))
                return np.array(bonus).sum(axis=1)
        return np.array([0,0])

    def _get_mismatch(self, motif, i, j, k, l, interior=True):
        """
        get bonus for interior mismatch where i is paired with j and k is
        mismatched with l
        """
        bp = get_base(motif[i, 0:4]) + get_base(motif[j, 0:4])
        mm = get_base(motif[k, 0:4]) + get_base(motif[l, 0:4])
        ix = s.mismatches.index(mm)*6 + s.bps.index(bp)
        if interior:
            return self.interior_mismatch[ix, :]
        return self.hp_mismatch[ix, :]

    def parse_nupack_params(self, filename, permute=False):
        """
        parse nn matrices from nupack parameter file
        return X with all motifs in matrix form where columns are
            A, C, G, U, (, ), 5' dangle, 3' dangle
            and y with energy for each motif
        """
        allX = {i: np.zeros((0, i, 8), dtype=bool) for i in range(2, 11)}
        ally = {i: np.zeros((0, 2)) for i in range(2, 11)}
        with open('%s.dG' % filename) as fG:
            with open('%s.dH' % filename) as fH:
                # parse stacking parameters
                X, y = self._parse_both_files(fG, fH, 'stacking')
                if permute:
                    X, y = self._get_all_circ_permutations(X, y)
                allX[4] = np.append(allX[4], X, 0)
                ally[4] = np.append(ally[4], y, 0)
                # parse base loop energies
                self._parse_base_energies_from_files(fG, fH)
                # parse hairpin loops
                # triloops
                fG.readline()
                fH.readline()
                X, y = self._parse_loops_from_files(fG, fH)
                if permute:
                    X, y = self._get_all_circ_permutations(X, y)
                allX[5] = np.append(allX[5], X, 0)
                ally[5] = np.append(ally[5], y, 0)
                # tetraloops
                X, y = self._parse_loops_from_files(fG, fH)
                if permute:
                    X, y = self._get_all_circ_permutations(X, y)
                allX[6] = np.append(allX[6], X, 0)
                ally[6] = np.append(ally[6], y, 0)
                # parse hp mismatches
                X, yG = self._parse_mismatches_from_file(fG)
                X, yH = self._parse_mismatches_from_file(fH)
                self.hp_mismatch = np.stack((yG, yH), axis=1)/100
                # parse interior mismatches
                fG.readline()
                fH.readline()
                X, yG = self._parse_mismatches_from_file(fG)
                X, yH = self._parse_mismatches_from_file(fH)
                self.interior_mismatch = np.stack((yG, yH), axis=1)/100
                # parse dangles
                X, y = self._parse_both_files(fG, fH, 'dangles')
                if permute:
                    X, y = self._get_all_circ_permutations(X, y)
                allX[3] = np.append(allX[3], X, 0)
                ally[3] = np.append(ally[3], y, 0)
                # parse multiloops
                X, y = self._parse_mloops_from_files(fG, fH, range(10))
                for l in X:
                    if permute:
                        X[l], y[l] = self._get_all_circ_permutations(X[l],
                                                                     y[l])
                    allX[l] = np.append(allX[l], X[l], 0)
                    ally[l] = np.append(ally[l], y[l], 0)
                # parse at penalty
                X, y = self._parse_AT_penalty(fG, fH)
                allX[2] = np.append(allX[2], X, 0)
                ally[2] = np.append(ally[2], y, 0)
                self.at_penalty /= 100
                # parse interior loops
                for _ in range(3):
                    X, y = self._parse_both_files(fG, fH, 'iloops')
                    l = X.shape[1]
                    if permute:
                        X, y = self._get_all_circ_permutations(X, y)
                    allX[l] = np.append(allX[l], X, 0)
                    ally[l] = np.append(ally[l], y, 0)
                self._parse_penalties_from_file(fG, fH)
        ally = {l: y/100 for l, y in ally.items()}
        return allX, ally

    def _get_all_circ_permutations(self, X, y):
        """
        get all circular permutations of each motif
        """
        n = X.shape
        fullX = np.zeros((n[0]*n[1], n[1], 8), dtype=bool)
        fully = np.zeros(n[0]*n[1])
        for i in range(n[1]):
            fullX[i*n[0]:(i+1)*n[0], :, :] = np.roll(X, i, axis=1)
            fully[i*n[0]:(i+1)*n[0]] = y
        return fullX, fully

    def _parse_both_files(self, fG, fH, motif):
        """ parse motif from both files and combine """
        if motif == 'stacking':
            func = self._parse_stacking_from_file
        elif motif == 'dangles':
            func = self._parse_dangles_from_file
        elif motif == 'iloops':
            func = self._parse_iloops_from_file

        XG, yG = func(fG)
        XH, yH = func(fH)
        assert np.all(XG == XH), (XG == XH)
        return XG, np.stack((yG, yH), axis=1)

    def _parse_stacking_from_file(self, f):
        """
        parse bp stacking parameters from file
        """
        f.readline()
        assert f.readline().startswith('>Stacking')
        for i in range(3):
            f.readline()
        X = self._get_stack_motifs()
        y = np.zeros(36)
        for i in range(6):
            y[i*6:(i+1)*6] = [float(e) for e in f.readline().split()]
        return X, y

    def _get_stack_motifs(self):
        """
        parse bp stacking parameters to nn matrix
        """
        X = np.zeros((36, 4, 8), dtype=bool)
        X[:, 0, 5] = 1
        X[:, 1, 4] = 1
        X[:, 2, 5] = 1
        X[:, 3, 4] = 1
        for i, bp1 in enumerate(s.bps):
            X[i*6:(i+1)*6, 0, s.bases[bp1[0]]] = 1
            X[i*6:(i+1)*6, 3, s.bases[bp1[1]]] = 1
            for j, bp2 in enumerate(s.bps):
                X[i*6+j, 1, s.bases[bp2[0]]] = 1
                X[i*6+j, 2, s.bases[bp2[1]]] = 1
        return X

    def _parse_base_energies_from_files(self, fG, fH):
        """
        parse loop base energies from files
        """
        hpG, bgG, ilG, asG = self._parse_base_energies_from_file(fG)
        hpH, bgH, ilH, asH = self._parse_base_energies_from_file(fH)
        self.hairpins = np.stack((hpG, hpH), axis=1)
        self.bulges = np.stack((bgG, bgH), axis=1)
        self.interiors = np.stack((ilG, ilH), axis=1)
        self.asymmetries = np.stack((asG, asH), axis=1)

    def _parse_base_energies_from_file(self, f):
        """
        parse loop base energies from file
        """
        assert f.readline().startswith('>Hairpin Loop Energies')
        hairpins = [float(e)/100 for e in f.readline().split()]
        assert f.readline().startswith('>Bulge loop Energies')
        bulges = [float(e)/100 for e in f.readline().split()]
        assert f.readline().startswith('>Interior Loop Energies')
        interiors = [float(e)/100 for e in f.readline().split()]
        assert f.readline().startswith('>NINIO asymmetry')
        for _ in range(3):
            f.readline()
        asymmetries = [float(e)/100 for e in f.readline().split()]
        return hairpins, bulges, interiors, asymmetries

    def _parse_penalties_from_file(self, fG, fH):
        """ parse polyc, pseudoknot, bimolecular penalties """
        assert fG.readline().startswith('>POLYC')
        assert fH.readline().startswith('>POLYC')
        for _ in range(3):
            fG.readline()
            fH.readline()
        self.polyc = np.stack(([float(e)/100 for e in fG.readline().split()],
                               [float(e)/100 for e in fH.readline().split()]),
                              axis=1)
        assert fG.readline().startswith('>BETA')
        assert fH.readline().startswith('>BETA')
        for _ in range(5):
            fG.readline()
            fH.readline()
        self.pk = np.stack(([float(e)/100 for e in fG.readline().split()],
                            [float(e)/100 for e in fH.readline().split()]),
                           axis=1)
        assert fG.readline().startswith('>BIMOLECULAR')
        assert fH.readline().startswith('>BIMOLECULAR')
        self.bimolecular = np.array([float(fG.readline()),
                                     float(fH.readline())])/100

    def _parse_loops_from_files(self, fG, fH):
        """
        parse loops from open nupack parameter file iterators,
        adding base energy to all parsed energies
        """
        loops = {}
        loops = self._parse_loops_from_file(loops, fG, 0)
        loops = self._parse_loops_from_file(loops, fH, 1)
        if len(loops) == 0:
            return np.zeros((0, 0), dtype=bool), np.zeros(0), line
        l = len(loops.iterkeys().next())
        base = self.hairpins[l-3]*100
        X = np.zeros((len(loops), l, 8), dtype=bool)
        y = np.zeros((len(loops), 2))
        for i, (loop, dG) in enumerate(loops.iteritems()):
            if len(loop) != l:
                raise ValueError('Input loops of unequal length')
            X[i, :, :] = self._parse_loop(loop)
            y[i, :] = dG + base
        return X, y
        
    def _parse_loops_from_file(self, loops, f, pos):
        """
        parse loops from open nupack parameter file iterator f,
        adding base energy to all parsed energies
        """
        line = f.readline()
        while line and not line.startswith('>'):
            fields = line.split()
            if fields[0] not in loops:
                loops[fields[0]] = np.zeros(2)
            loops[fields[0]][pos] = float(fields[1])
            line = f.readline()
        return loops

    def _parse_loop(self, seq):
        """
        parse loop sequence to nn matrix
        """
        mat = np.zeros((len(seq), 8), dtype=bool)
        mat[0, 5] = 1
        mat[-1, 4] = 1
        for i, char in enumerate(seq):
            mat[i, s.bases[char]] = 1
        return mat

    def _parse_mismatches_from_file(self, f):
        """
        parse mismatch parameters from file
        """
        for _ in range(2):
            f.readline()
        X = np.zeros((96, 4, 8), dtype=bool)
        y = np.zeros(96)

        for i, mismatch in enumerate(s.mismatches):
            X[i*6:(i+1)*6, 0, s.bases[mismatch[0]]] = 1
            X[i*6:(i+1)*6, 3, s.bases[mismatch[1]]] = 1
            y[i*6:(i+1)*6] = [float(e) for e in f.readline().split()]
            for j, pair in enumerate(s.bps):
                X[i*6+j, 1, s.bases[pair[0]]] = 1
                X[i*6+j, 2, s.bases[pair[1]]] = 1
                X[i*6+j, 1, 4] = 1
                X[i*6+j, 2, 5] = 1
        return X, y

    def _parse_dangles_from_file(self, f):
        """
        parse dangle parameters from file
        """
        X = np.zeros((48, 3, 8), dtype=bool)
        y = np.zeros(48)

        # parse 3' dangles
        line = f.readline()
        assert line.startswith('>Dangle Energies'), line
        for _ in range(3):
            f.readline()
        X[:24, :, :], y[:24] = self._get_dangle_motifs(f, five=0)

        # parse 5' dangles
        line = f.readline()
        assert line.startswith('>Dangle Energies'), line
        for _ in range(3):
            f.readline()
        X[24:, :, :], y[24:] = self._get_dangle_motifs(f, five=1)

        return X, y

    def _get_dangle_motifs(self, f, five):
        """
        parse one set of dangles from file
        five is boolean indicating whether dangle is 5'
        """
        five = int(five)
        X = np.zeros((24, 3, 8), dtype=bool)
        y = np.zeros(24)

        for i, pair in enumerate(s.bps):
            X[i*4:(i+1)*4, five+1, s.bases[pair[0]]] = 1
            X[i*4:(i+1)*4, five, s.bases[pair[1]]] = 1
            X[i*4:(i+1)*4, five, 4] = 1
            X[i*4:(i+1)*4, five+1, 5] = 1
            for j, dangle in enumerate(s.bases):
                X[i*4+j, 2*(1-five), s.bases[dangle]] = 1
                X[i*4+j, 2*(1-five), 7-five] = 1
            if f:
                y[i*4:(i+1)*4] = [float(e) for e in f.readline().split()]
        return X, y

    def _parse_mloops_from_files(self, fG, fH, ls):
        """
        parse multiloop parameters from files and generate all multiloops with
        lengths in ls
        """
        assert fG.readline().startswith('>Multiloop terms')
        assert fH.readline().startswith('>Multiloop terms')
        for _ in range(2):
            fG.readline()
            fH.readline()
        self.mloop_params = np.stack(
            ([float(e)/100 for e in fG.readline().split()],
             [float(e)/100 for e in fH.readline().split()]), axis=1)
        ls = [l for l in ls if l >= 6]  # cannot have multiloop with length < 6
        X, y = self._generate_rand_mloops(self.mloop_params, ls)
        return X, y

    def _parse_AT_penalty(self, fG, fH):
        """ parse AT penalty from file """
        assert fG.readline().startswith('>AT_PENALTY')
        assert fH.readline().startswith('>AT_PENALTY')
        fG.readline()
        fH.readline()
        self.at_penalty = np.array([float(fG.readline()), float(fH.readline())])
        X = np.zeros((4, 2, 8), dtype=bool)
        X[0, 0, s.bases['A']] = 1
        X[0, 1, s.bases[s.baselist[-1]]] = 1 # U or T
        X[1, 0, s.bases[s.baselist[-1]]] = 1
        X[1, 1, s.bases['A']] = 1
        X[2, 0, s.bases['G']] = 1
        X[2, 1, s.bases[s.baselist[-1]]] = 1
        X[3, 0, s.bases[s.baselist[-1]]] = 1
        X[3, 1, s.bases['G']] = 1
        X[:, 0, 4] = 1
        X[:, 1, 5] = 1
        return X, np.tile(self.at_penalty, (4,1))

    def _generate_rand_mloops(self, params, ls, n=1000):
        """
        generate n random mloops for each length & number of stems
        """
        X = {}
        y = {}
        # loop over possible lengths
        for l in ls:
            nstems = range(3, l/2)
            X[l] = np.zeros((n*len(nstems), l, 8), dtype=bool)
            y[l] = np.zeros((n*len(nstems), 2))
            # loop over possible stem number
            for i, p in enumerate(nstems):
                u = l-2*p  # number of unpaired bases
                y[l][i*n:(i+1)*n] = params[0] + p*params[1] + \
                    u*params[2]
                for j in range(n):
                    paired, unpaired = self._generate_pair_positions(p, u)
                    # random unpaired bases
                    ru = np.random.randint(4, size=u)
                    X[l][i*n+j, unpaired, ru] = 1
                    # random base pairs
                    rs = [s.bps[r] for r in np.random.randint(6, size=p)]
                    X[l][i*n+j, paired, [s.bases[r[0]] for r in rs]] = 1
                    X[l][i*n+j, paired+1, [s.bases[r[1]] for r in rs]] = 1
        return X, y

    def _generate_pair_positions(self, s, p):
        """
        randomly generate positions of pairs and unpaired bases in multiloop
        s = number of stems
        u = number of unpaired bases
        """
        r = np.random.choice(s + p, s, replace=False)
        paired = []
        unpaired = []
        i = 0
        for j in range(s + p):
            if j in r:
                paired.append(i)
                i += 2
            else:
                unpaired.append(i)
                i += 1
        return np.array(paired), np.array(unpaired)

    def _parse_iloops_from_file(self, f):
        """
        parse interior loops from open nupack parameter file f,
        adding base energy and asymmetry energy to all parsed energies
        """
        assert f.readline().startswith('>Interior Loops')
        energies = {}
        for _ in range(4):
            f.readline()
        line = f.readline()
        m, n = self._get_iloop_size(line.strip())
        for i in range(36*4**(m+n-2)):
            if i != 0:
                line = f.readline()
            loop = line.strip()
            energies[loop] = f.readline().split()
            energies[loop].extend(f.readline().split())
            energies[loop].extend(f.readline().split())
            energies[loop].extend(f.readline().split())
            energies[loop] = np.array(energies[loop])
        X, y = self._parse_iloops(energies)
        # y += self._get_iloop_energy(m, n, asym=False)
        return X, y

    def _get_iloop_energy(self, m, n, asym=True):
        """
        get base energy of internal loop
        """
        base = self.interiors[min(m+n-1, 29), :]
        if asym:
            asym_dG = abs(m-n) * self.asymmetries[min(4, m, n)-1, :]
            if self.asymmetries[4, 0] <= asym_dG[0]:
                return base + self.asymmetries[4, :]
            else:
                return base + asym_dG
        else:
            return base

    def _parse_iloops(self, energies):
        """
        parse mxn internal loops to nn matrices

        args:
            energies: dictionary with key as string parameterizing fixed loop
                      elements as follows

                      CG..AU = 5'- C X A 3'
                               3'- G Y U 5'
                      CG.A..AU = 5'- C A   A -3'
                                 3'- G Y X U -5'
                      CG.AG..AU = 5'- C A G A -3'
                                  3'- G Y X U -5'

                      and value as 4x4 matrix of energies as follows

                      rows: X = A C G U (X constant for a row)
                      columns: Y = A C G U (Y constant in column)
        returns:
            X, y: where X is the 3d array of nn matrices and
                  y is the 1d array of energies
        """
        m, n = self._get_iloop_size(energies.keys()[0])
        X = np.zeros((16*len(energies), 4+m+n, 8), dtype=bool)
        y = np.zeros(16*len(energies))
        i = 0
        for loop, values in energies.iteritems():
            X[i*16:(i+1)*16, :, :] = self._get_iloop_motifs(m, n, loop)
            y[i*16:(i+1)*16] = values.flatten()
            i += 1
        return X, y

    def _get_iloop_motifs(self, m, n, loop):
        """ get matrix of internal loop motifs """
        mat = self._get_mxn_iloop_motifs(m, n, loop)
        fullmat = np.tile(mat, (16, 1, 1))
        fullmat[:, m+3*n-3, 0:4] = np.repeat(np.identity(4), 4, 0)
        fullmat[:, m+n+2, 0:4] = np.tile(np.identity(4), (4, 1))
        return fullmat

    def _get_iloop_size(self, string):
        """
        get internal loop dimensions from string represententation

        args:
            s: string representation of internal loop as follows
        returns:
            dimensions of internal loop mxn
        """
        bases = ''.join(s.baselist)
        if re.match(r'[%s]{2}\.\.[%s]{2}' % tuple([bases]*2), string):
            return 1, 1
        elif re.match(r'[%s]{2}\.[%s]\.\.[%s]{2}' % tuple([bases]*3), string):
            return 1, 2
        elif re.match(r'[%s]{2}\.[%s]{2}\.\.[%s]{2}' % tuple([bases]*3), string):
            return 2, 2
        raise ValueError('Invalid internal loop string %s' % string)

    def _check_iloop_string(self, m, n, string):
        """
        check that string representation matches given internal loop dimensions
        """
        if m > 2 or n > 2 or m > n:
            raise ValueError('Invalid internal loop %dx%d' % (m, n))

        bases = ''.join(s.baselist)
        if m == 1 and n == 1:
            return re.match(r'[%s]{2}\.\.[%s]{2}' % tuple([bases]*2), string)
        if m == 1 and n == 2:
            return re.match(r'[%s]{2}\.[%s]\.\.[%s]{2}' % tuple([bases]*3), string)
        if m == 2 and n == 2:
            return re.match(r'[%s]{2}\.[%s]{2}\.\.[%s]{2}' % tuple([bases]*3), string)

        assert False, 'No iloop matches in check_iloop_string'

    def _get_mxn_iloop_motifs(self, m, n, str_):
        """
        parse string representing an mxn internal loop into nn matrix

        args:
            m: the length of one side of the internal loop
            n: the length of the other side of the internal loop
            s: string representation of internal loop (see parse_iloop)
        """
        if m > 2 or n > 2 or m > n:
            raise ValueError('Invalid internal loop %dx%d' % (m, n))
        if not self._check_iloop_string(m, n, str_):
            raise ValueError('Invalid internal loop string %s for %dx%d'
                             % (str_, m, n))

        # initialize matrix
        looplen = m+n
        mat = np.zeros((4+looplen, 8), dtype=bool)

        # set base identity indicators
        mat[0, s.bases[str_[0]]] = 1
        mat[1+m, s.bases[str_[1+m+2*n]]] = 1
        mat[2+m, s.bases[str_[2+m+2*n]]] = 1
        mat[3+looplen, s.bases[str_[1]]] = 1
        if n > 1:
            mat[1, s.bases[str_[3]]] = 1
            if m > 1:
                mat[2, s.bases[str_[4]]] = 1

        # set pairing indicators
        mat[0, 5] = 1
        mat[-1, 4] = 1
        mat[1+m, 4] = 1
        mat[2+m, 5] = 1
        return mat

    def to_file(self, filename):
        """ write to params files """
        self.to_file_pos('%s.dG' % filename, 0)
        self.to_file_pos('%s.dH' % filename, 1)
        
    def to_file_pos(self, filename, pos):
        """ write one params file """
        with open(filename, 'w') as f:
            # stacking
            f.write('>Stacking\n')
            X = self._get_stack_motifs()
            n = len(s.bps)
            for i in range(n):
                for j in range(n):
                    f.write(self._format_energy(self.find_matching_motif(
                        X[i*n+j, :, :])[pos]))
                f.write('\n')
            # hairpin loop
            f.write('>Hairpin Loop\n')
            f.write(''.join([self._format_energy(e[pos]) for e in self.hairpins]))
            f.write('\n')
            # bulge loop
            f.write('>Bulge Loop\n')
            f.write(''.join([self._format_energy(e[pos]) for e in self.bulges]))
            f.write('\n')
            # interior loop
            f.write('>Interior Loop\n')
            f.write(''.join([self._format_energy(e[pos]) for e in self.interiors]))
            f.write('\n')
            # asymmetry
            f.write('>NINIO asymmetry\n')
            f.write(''.join([self._format_energy(e[pos]) for e in
                             self.asymmetries]))
            f.write('\n')
            # loops
            f.write('>Triloops\n')
            X = self._get_nloop_motifs(3)
            for i in range(X.shape[0]):
                seq = [s.baselist[np.where(X[i, j, 0:4])[0][0]] for j in
                       range(5)]
                e = self.find_matching_motif(X[i, :, :], bonus=False)[pos] - \
                    self.hairpins[2][pos]
                f.write('%s\t%s' % (''.join(seq), self._format_energy(e)))
                f.write('\n')
            f.write('>Tetraloops\n')
            X = self._get_nloop_motifs(4)
            for i in range(X.shape[0]):
                seq = [s.baselist[np.where(X[i, j, 0:4])[0][0]] for j in
                       range(6)]
                e = self.find_matching_motif(X[i, :, :], bonus=False)[pos] - \
                    self.hairpins[3][pos]
                f.write('%s\t%s' % (''.join(seq), self._format_energy(e)))
                f.write('\n')
            # hp mismatch
            f.write('>Mismatch HP\n')
            n = len(s.bps)
            for i in range(len(s.baselist)**2):
                for j in range(n):
                    f.write(self._format_energy(self.hp_mismatch[i*n+j][pos]))
                f.write('\n')
            # interior mismatch
            f.write('>Mismatch Interior\n')
            n = len(s.bps)
            for i in range(len(s.baselist)**2):
                for j in range(n):
                    f.write(self._format_energy(self.interior_mismatch[i*n+j][pos]))
                f.write('\n')
            # 3' dangles
            f.write('>Dangle Energies 3\'\n')
            X, _ = self._get_dangle_motifs(None, 0)
            n = len(s.baselist)
            for i in range(len(s.bps)):
                for j in range(n):
                    f.write(self._format_energy(self.find_matching_motif(
                        X[i*n+j, :, :])[pos]))
                f.write('\n')
            # 5' dangles
            f.write('>Dangle Energies 5\'\n')
            X, _ = self._get_dangle_motifs(None, 1)
            n = len(s.baselist)
            for i in range(len(s.bps)):
                for j in range(n):
                    f.write(self._format_energy(self.find_matching_motif(
                        X[i*n+j, :, :])[pos]))
                f.write('\n')
            # multiloop terms
            f.write('>Multiloop terms\n')
            f.write(''.join([self._format_energy(e[pos]) for e in
                             self.mloop_params]))
            f.write('\n')
            # at penalty
            f.write('>AT_PENALTY:\n')
            f.write(self._format_energy(self.at_penalty[pos]))
            f.write('\n')
            # 1x1 interior loops
            f.write('>Interior Loops 1x1\n')
            n = len(s.baselist)
            for bp1 in s.bps:
                for bp2 in s.bps:
                    str_ = '%s..%s' % (bp1, bp2)
                    f.write('%s\n' % str_)
                    X = self._get_iloop_motifs(1, 1, str_)
                    for i in range(n):
                        for j in range(n):
                            f.write(self._format_energy(
                                self.find_matching_motif(X[i*n+j, :, :])[pos]))
                        f.write('\n')
            # 2x2 interior loops
            f.write('>Interior Loops 2x2\n')
            for bp1 in s.bps:
                for bp2 in s.bps:
                    for b1 in s.baselist:
                        for b2 in s.baselist:
                            str_ = '%s.%s%s..%s' % (bp1, b1, b2, bp2)
                            f.write('%s\n' % str_)
                            X = self._get_iloop_motifs(2, 2, str_)
                            for i in range(n):
                                for j in range(n):
                                    f.write(self._format_energy(
                                        self.find_matching_motif(
                                            X[i*n+j, :, :])[pos]))
                                f.write('\n')
            # 1x2 interior loops
            f.write('>Interior Loops 1x2\n')
            for bp1 in s.bps:
                for bp2 in s.bps:
                    for b1 in s.baselist:
                        str_ = '%s.%s..%s' % (bp1, b1, bp2)
                        f.write('%s\n' % str_)
                        X = self._get_iloop_motifs(1, 2, str_)
                        for i in range(n):
                            for j in range(n):
                                f.write(self._format_energy(
                                    self.find_matching_motif(X[i*n+j, :, :])[pos]))
                            f.write('\n')
            # poly C
            f.write('>POLYC\n')
            f.write(''.join([self._format_energy(e[pos]) for e in
                             self.polyc]))
            f.write('\n')
            # pseudoknot energies
            f.write('>BETA\n')
            f.write(''.join([self._format_energy(e[pos]) for e in
                             self.pk]))
            f.write('\n')
            # bimolecular
            f.write('>BIMOLECULAR\n')
            f.write(self._format_energy(self.bimolecular[pos]))
            f.write('\n')

    def _format_energy(self, energy, n=5):
        """ reformat energy as string with left padding """
        return ('%.0f' % (energy*100)).rjust(n)

    def _get_nloop_motifs(self, n):
        """ get motifs that are size n hairpin loops """
        if n not in [3, 4]:
            raise ValueError('only triloops and tetraloops are stored '
                             'explicitly')
        X = self.X[n+2]
        nmotifs = X.shape[0]
        return X[np.sum(X[:, :, 4], axis=1) == 1, :, :]


def get_base(vec):
    """ get base identity from one hot vector """
    if np.sum(vec) != 1:
        raise ValueError('invalid one hot notation')
    try:
        return s.baselist[np.where(vec)[0][0]]
    except:
        raise ValueError('invalid one hot base vector')


def get_iloop_lens(motif):
    """ get length of each side of internal loop """
    pos = np.where(motif[:, 4])[0][0]
    assert motif[pos+1, 5] == 1, 'must form pair'
    m = pos-1
    n = motif.shape[0]-m-4
    return pos, min(m, n), max(m, n)
