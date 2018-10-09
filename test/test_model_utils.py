import unittest as ut
from ribonet.model_utils import *
import ribonet.params as pp
import ribonet.ensemble as e
import ribonet.settings as s
import tensorflow as tf
import copy


class modelUtilsUT(ut.TestCase):


    def setUp(self):
        tf.reset_default_graph()
        s.set_molecule('dna')

    def test_ribonet_trainer(self):
        d = e.MeltDataset(['%s/test/files/testdna.melt' % s.BASE_DIR],
                          pickle=False)
        t = RiboNetTrainer(d, '5', optimizer='descent')

        motif = t.data.data.ensemble[0].motifs_arrays[4][0, :, :]
        preweight = t.get_weights(0)
        preloss = t.get_loss()
        t.restore('%s/test/files/model' % s.BASE_DIR)
        assert not np.all(preweight == t.get_weights(0))
        assert preloss != t.get_loss()
        t.train(1, 2)
        t.get_loss()

        tf.reset_default_graph()
        t = RiboNetTrainer(d, '5', optimizer='adagrad')
        with self.assertRaises(ValueError):
            t = RiboNetTrainer(d, '5', optimizer='invalid')

    def test_melt_trainer(self):
        d = e.dGDataset('%s/test/files/test.dG' % s.BASE_DIR)
        t = MeltTrainer(d, '5')

if __name__ == '__main__':
    ut.main()
