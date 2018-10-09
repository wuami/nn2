import unittest as ut
import numpy as np
from ribonet.ensemble import *
import ribonet.params as pp
import ribonet.settings as s

class ensembleUT(ut.TestCase):

    def setUp(self):
        s.set_molecule('rna')
        self.p = pp.NNParams('%s/test/files/rna1995' % s.BASE_DIR)

    def check_motif_dict(self, d1, d2):
        self.assertEqual(d1.keys(), d2.keys())
        for key in d1.keys():
            try:
                np.testing.assert_almost_equal(d1[key], d2[key])
            except AssertionError as e:
                print d1[key]
                print d2[key]
                raise AssertionError('arrays for key %d do not match\n%s'
                                     % (key, e))

    def test_get_motifs(self):
        motifs = {4: np.array([[[0, 1, 0, 0, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 1, 0, 0, 0]]])}
        self.check_motif_dict(get_motifs('CCGG', '(..)')[0], motifs)

        motifs = {2:  np.array([[[1, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1, 0, 0]]]),
                  4: np.array([[[1, 0, 0, 0, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0, 0]]])}
        self.check_motif_dict(get_motifs('ACGU', '(..)')[0], motifs)

        motifs[3] = np.array([[[1, 0, 0, 0, 0, 0, 1, 0],
                               [1, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 1, 0, 0]]])
        self.check_motif_dict(get_motifs('AACGU', '.(..)')[0], motifs)

        motifs[3] = np.array([[[1, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 1]]])
        self.check_motif_dict(get_motifs('ACGUU', '(..).')[0], motifs)

        with self.assertRaises(ValueError):
            get_motifs('AACGU', '.(..).')

        motifs[3] = np.array([[[1, 0, 0, 0, 0, 0, 1, 0],
                               [1, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 1, 0, 0]],
                              [[1, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 1]]])
        self.check_motif_dict(get_motifs('AACGUG', '.(..).')[0], motifs)

        motifs = {2: np.array([[[1, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1, 0, 0]]]),
                  3: np.array([[[1, 0, 0, 0, 0, 1, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0, 0]],
                               [[1, 0, 0, 0, 0, 1, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0, 0]],
                               [[1, 0, 0, 0, 0, 0, 1, 0],
                                [1, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1, 0, 0]],
                               [[1, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 1]]]),
                  9: np.array([[[1, 0, 0, 0, 0, 1, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0, 0]]])}

        self.check_motif_dict(get_motifs('AAAAAUAAAUAUA', '.(.(.).(.).).')[0],
                              motifs)

        motifs, alts, meltmotifs, meltalt = get_motifs('AAUAUAA', '(.).(.)')
        self.assertEqual(alts, [[[(3, 3), (3, 2)]]])
        self.assertEqual(len(motifs[2]), 2)
        self.assertEqual(len(motifs[3]), 4)

        # test AU penalty parsing
        self.assertEqual(get_motifs('AUAGUGACGCCUGCAUAUAU',
                                    '((.(((.......)))))..')[0][2].shape[0], 1)
        self.assertEqual(get_motifs('UCUCGCACGUACGGAAUACC',
                                    '(.(((......))).)....')[0][2].shape[0], 1)
        self.assertEqual(get_motifs('GCCGGACCUGCGCCGUUCCC',
                                    '...(((...((...))))).')[0][2].shape[0], 2)

        # test motifs with meltstruct
        motifs, alts, meltmotifs, meltalts = get_motifs('CCACCAAAAGGGG', '((.((....))))', '...((....))..')
        self.assertEqual(sum([len(m) for m in motifs.values()]), 2)

    def test_motif(self):
        motif = np.array([[1, 0, 0, 0, 0, 0, 1, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 1, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 1]])
        m = EnsembleMotif(motif, 2)
        m.add_conformation(1)
        np.testing.assert_almost_equal(m.conformations, np.array([0, 1]))

    def test_conformation(self):
        # test basic usage
        c = Conformation('GCGAAAGC', 1.0, '((....))')
        self.assertFalse(c.forms_ms2())
        self.assertFalse(c.forms_fmn())

        with self.assertRaises(ValueError):
            Conformation('GCGAAAGC', 1.0, '.')

        # test for ms2 detection
        c = Conformation('ACAUGAGGAUCACCCAUGU', 1.0, '(((((.((....)))))))')
        self.assertTrue(c.forms_ms2())
        self.assertFalse(c.forms_fmn())

        c = Conformation('ACAUGAGGAUCACCCAUGUGGGGGAGGAAAACCCCCCC', 1.0,
                         '...................(((((.((....)))))))')
        self.assertFalse(c.forms_ms2())
        self.assertFalse(c.forms_fmn())

        # test for fmn detection
        c = Conformation('AAAAGGAUAUAAAUUAGAAGGUUU', 1.0,
                         '(((......(((.))).....)))')
        self.assertFalse(c.forms_ms2())
        self.assertTrue(c.forms_fmn())

        c = Conformation('UUAGAAGGAAAAUUUAGGAUAUAA', 1.0,
                         '(((.....(((.)))......)))')
        self.assertFalse(c.forms_ms2())
        self.assertTrue(c.forms_fmn())

        c = Conformation('UUAGAAGGAAAAAUUUGGAUAUAA', 1.0,
                         '(((......(((.))).....)))')
        self.assertFalse(c.forms_ms2())
        self.assertFalse(c.forms_fmn())

        c = Conformation('AAAAUUAGAAGGAAAAGAAGGAAAAUUUAGGAUAUAA', 1.0,
                         '....(((.....(((..........)))......)))')
        self.assertFalse(c.forms_ms2())
        self.assertTrue(c.forms_fmn())

        # check alternate motif energy calculations
        p2 = pp.NNParams('%s/test/files/alt' % s.BASE_DIR, molecule='rna')
        c = Conformation('AAAUAAUA', 0, '.(.)(.).')
        np.testing.assert_allclose(c.get_energy(self.p), -2.7)
        np.testing.assert_allclose(c.get_energy(p2), -0.6)

    def test_ensemble(self):
        e = Ensemble('GCGAAAGC', params='rna1995')
        
        self.assertEqual(len(e.conformations), 3)
        self.assertEqual([len(c.motifs) for c in e.conformations], [2, 0, 2])
        self.assertEqual(len(e.motifs), 4)
        motif = np.array([[0, 1, 0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 1, 0, 0, 0]])
        hash_ = pp.get_np_hash(motif)
        self.assertTrue(hash_ in e.motifs)
        grad = e.get_grad_kd(motif, 0)
        self.assertEqual(grad, 0)

        # check adjusting of motif energies
        before = np.copy(e.dG)
        e.adjust_motif_energy(motif, -0.1)
        np.testing.assert_allclose(before - e.dG,
                                   np.array([0.1, 0, 0.1]))

        motif = np.array([[0, 0, 1, 0, 0, 0, 1, 0],
                          [0, 1, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 1, 0, 0]])
        before = np.copy(e.dG)
        e.adjust_motif_energy(motif, -0.1)
        np.testing.assert_allclose(before - e.dG,
                                   np.array([0, 0, 0.1]))


        #e = Ensemble('ACAUGAGGAUCACCCAUGU', params='rna1995')
        #for c in e.conformations:
        #    np.testing.assert_approx_equal(c.dG, c.get_energy(self.p))

        # numerical checks
        e = Ensemble(('AUCUUGUCAGGAUAUCGUACAAUAUGGAAAACAUAGAGGUACGAGAAGGGACAUG'
                      'AGGAUCACCCAUGUAAAAC'))
        # check pf calculations
        pf_kd = e.get_pf()/e.get_pf(e.ms2)*kd_ms2
        np.testing.assert_approx_equal(e.get_predicted_kd(0), pf_kd)

    def test_predict_kd(self):
        np.testing.assert_approx_equal(predict_kd(-10, -5, -5, -10, 0),
                                       2e-08)
        np.testing.assert_approx_equal(predict_kd(-2.1, -3.2, -4.8, 10,
                                                  20),
                                       2.8320840244339147e-06)
        self.assertTrue(np.isnan(predict_kd(np.inf, np.inf, np.inf, np.inf,
                                            0)))


    def test_gradient_check(self):
        # get one sequence & one motif from sequence
        e = Ensemble('GCGAAAGC', params=self.p)
        motif = np.array([[0, 0, 1, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 1, 0, 0, 0]])
        delta = 1e-5
        
        # set up energies
        e.update_energies(self.p)

        # get gradients
        grad_melt = e.get_grad_melt(motif, 37)

        dG = self.p.get_energy(motif)
        # dG + delta
        self.p.update_energy(motif, dG + delta)
        e.update_energies(self.p)
        np.testing.assert_almost_equal(dG + delta, self.p.get_energy(motif))
        dGs_high = e.dG
        melt_high = e.get_predicted_melted(37)
        # dG - delta
        self.p.update_energy(motif, dG - delta)
        e.update_energies(self.p)
        dGs_low = e.dG
        melt_low = e.get_predicted_melted(37)

        # check conformation energies
        dGs_diff = dGs_high - dGs_low
        self.assertTrue(np.all((dGs_diff == 0) | np.isclose(dGs_diff, 2*delta)))
        
        # check gradients
        np.testing.assert_almost_equal(melt_high-melt_low, grad_melt*2*delta)
        
    def test_melt_dataset(self):
        # test initialization
        m = MeltDataset(['%s/test/files/test.melt' % s.BASE_DIR],
                        pickle=False)
        motif = np.array([[0, 0, 0, 1, 0, 1, 0, 0],
                          [0, 1, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 1, 0, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0]])
        self.melt_grad_check(m, motif)    

    def melt_grad_check(self, m, motif, delta=1e-10):
        # check gradients for one motif
        m.update_energies(self.p)

        grad_loss = m.get_grad_loss(motif)

        # dG + delta
        dG = self.p.get_energy(motif)
        self.p.update_energy(motif, dG + delta)
        m.update_energies(self.p)
        m.get_predictions()
        loss_high = m.get_rmse()**2 * m.n
        # dG - delta
        self.p.update_energy(motif, dG - delta)
        m.update_energies(self.p)
        m.get_predictions()
        loss_low = m.get_rmse()**2 * m.n
        
        # check gradients
        np.testing.assert_almost_equal(loss_high-loss_low,
                                       grad_loss * delta * 2, 6)

    def test_dg_dataset(self):
        # test initialization
        d = dGDataset('%s/test/files/test.dG' % s.BASE_DIR)
        for l in d.get_motif_lengths():
            motifs = d.get_motif_array(l)
            self.p.add_motifs(motifs, np.zeros((motifs.shape[0], 2)))

        motif = np.array([[0, 0, 1, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 1, 0, 0, 0]])
        self.dg_grad_check(d, motif)
        
        motif = np.array([[0, 0, 0, 1, 0, 1, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 1, 0, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0]])
        self.dg_grad_check(d, motif)
        
        motif = np.array([[0, 0, 0, 1, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0]])
        self.dg_grad_check(d, motif)
    
    def dg_grad_check(self, d, motif, delta=1e-10):
        d.update_energies(self.p)
        d.get_predictions()

        grad_loss = d.get_grad_loss(motif)

        # dG + delta
        dG = self.p.get_energy(motif)
        self.p.update_energy(motif, dG + delta)
        d.update_energies(self.p)
        loss_high = d.get_rmse()**2 * d.n
        # dG - delta
        self.p.update_energy(motif, dG - delta)
        d.update_energies(self.p)
        loss_low = d.get_rmse()**2 * d.n
        
        np.testing.assert_almost_equal(loss_high-loss_low,
                                       grad_loss * delta * 2, 6)


if __name__ == '__main__':
    ut.main()
