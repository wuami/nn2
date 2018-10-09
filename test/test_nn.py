import unittest as ut
from ribonet.nn import *
import ribonet.params as pp
import ribonet.settings as s
import numpy as np
import tensorflow as tf


class NNUT(ut.TestCase):

    test_file = '%s/dna1998' % s.PARAMS_DIR

    def setUp(self):
        tf.reset_default_graph()

    def test_augment(self):
        t = tf.placeholder('float', [1, 3, 1, 1])
        t = augment(t)
        tf.reset_default_graph()

    def test_create_conv_layer(self):
        t = tf.ones([3, 1, 2, 4])
        layer = create_conv_layer(t, 4, 4)
        tf.reset_default_graph()

    def test_cnn(self):
        cnn = CNN('5,1', low_mem=True)
        n = pp.NNParams(self.test_file, split=True)
        cnn.train(n, 1)
        for i in range(5, 9):
            X_test, y_test = n.get_test(i)
            cnn.test(X_test)

        with self.assertRaises(ValueError):
            cnn.get_activation_profile(1, 0)

        cnn.init_writer()
        cnn.get_activation_profile(1, 0)

        cnn.get_weights(1)
        tf.reset_default_graph()

        cnn = CNN('5,1', optimizer='adagrad')
        with self.assertRaises(ValueError):
            cnn = CNN('5,1', optimizer='invalid')

            
if __name__ == '__main__':
    ut.main()
