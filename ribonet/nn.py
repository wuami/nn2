import tensorflow as tf
import numpy as np
import string
import random
import settings as s
import warnings
import sys
import os

MAXCHUNK = 10000

def augment(t, n=2):
    """ augment tensor to include copy of first n rows at bottom """
    top = tf.slice(t, [0, 0, 0, 0], [-1, n, -1, -1])
    return tf.concat([t, top], 1, name='cyclize')

def augment_recursive(t, n=2):
    """ augment tensor to include copy of first n rows at bottom,
    apply recursion if n is greater than size of tensor
    """
    return tf.cond(n > tf.shape(t)[1],
                   lambda: augment(augment(t, n/2), n/2 + n%2),
                   lambda: augment(t, n))

def weight_variable(shape, var='scaled', name='weights'):
    """
    create weight variable with random initialization
    either fixed variance of 0.1 or scaled to sqrt(2/n) where n is number of
        inputs

    Args:
        shape (array): dimensions of weight tensor
        var (float): fixed or scaled variance
        name (str): name of variable

    Returns:
        tf.Variable: weight variable

    Raises:
        ValueError: if invalid value is specified for "var"
    """
    with tf.device('/cpu:0'):
        if var == 'fixed':
            v = tf.get_variable(
                name, shape,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        elif var == 'scaled':
            v = tf.get_variable(
                name, shape,
                initializer=tf.truncated_normal_initializer(
                    mean=0, stddev=np.sqrt(2./np.prod(shape[0:3]))))
        else:
            raise ValueError('var must be either \'fixed\' or \'scaled\'')
    return v


def bias_variable(shape, name='biases'):
    """
    create bias variable with constant initialization

    Args:
        shape (array): dimensions of bias tensor
        name (str): name of variable

    Returns:
        tf.Variable: bias variable
    """
    with tf.device('/cpu:0'):
        v = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))
    return v


def conv2d(x, W, name='conv2d', dilation=1):
    """
    create 2d convolution

    Args:
        x (tf.Tensor): input to convolutions
        W (tf.Tensor): weights for convolutions
        name (str): name of operation

    Returns:
        tf.Tensor: resulting tensor after convolutions are performed
    """
    return tf.nn.convolution(x, W, strides=[1, 1], padding='VALID',
                             dilation_rate=[dilation, 1], name=name)


def batch_norm(x, depth):
    """perform batch normalization
    
    Args:
        x (tf.Tensor): input tensor to normalize
        depth (int): length of dimension to normalize over
    
    Returns:
        tf.Tensor: normalized tensor
    """
    mean = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False)
    variance = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)
    beta = tf.Variable(tf.constant(0.0, shape=[depth]))
    gamma = tf.Variable(tf.constant(1.0, shape=[depth]))

    batch_mean, batch_variance = tf.nn.moments(x, [0, 1, 2])
    assign_mean = mean.assign(batch_mean)
    assign_variance = variance.assign(batch_variance)
    with tf.control_dependencies([assign_mean, assign_variance]):
        return tf.nn.batch_norm_with_global_normalization(
            x, mean, variance, beta, gamma, 1e-4, True, name='batch_norm')


def create_conv_layer(input_, n_in, n_out, name='conv', bias=True,
                      norm=False, keep_prob=1):
    """
    create layer with tensor augmentation, convolution and relu

    Args:
        input_ (tf.Tensor): input to convolutional layer
        n_in (int): number of dimensions in input data
        n_out (int): number of dimensions (i.e. filters) in output
        name (str): name of layer
        bias (bool): whether or not to include bias term
        norm (bool): whether or not to apply batch normalization
        keep_prob (float): probability to keep node

    Returns:
        tf.Tensor, tf.Tensor: weights, outputs
    """
    with tf.variable_scope(name):
        W = weight_variable([3, 1, n_in, n_out])

        input_ = tf.nn.dropout(input_, keep_prob)
        input_ = augment_recursive(input_)
        conv = conv2d(input_, W)
        if bias:
            b = bias_variable([n_out])
            conv = conv + b
        if norm:
            relu = tf.nn.relu(batch_norm(conv, n_out), name='relu')
        else:
            relu = tf.nn.relu(conv, name='relu')
    return W, relu
    

def create_dilated_conv_layer(input_, n_in, n_out, name='dilated_conv',
                              bias=True, norm=False, keep_prob=1):
    """
    create layer with tensor augmentation, dilated convolutions and relu

    Args:
        input_ (tf.Tensor): input to convolutional layer
        n_in (int): number of dimensions in input data
        n_out (int): number of dimensions (i.e. filters) in output
        name (str): name of layer
        bias (bool): whether or not to include bias term
        norm (bool): whether or not to apply batch normalization
        keep_prob (float): probability to keep node

    Returns:
        tf.Tensor, tf.Tensor: weights, outputs
    """
    n_perdil = [n_out/2] * 2
    n_perdil[0] = n_out - n_perdil[0]
    with tf.variable_scope(name):
        weights = []
        convs = []
        input_ = tf.nn.dropout(input_, keep_prob)
        for i, dilation in enumerate([1, 2]):
            W = weight_variable([3, 1, n_in, n_perdil[i]],
                                name='w_dilute%d' % dilation)

            conv = conv2d(augment_recursive(input_, 2 * dilation), W,
                          dilation=dilation)
            weights.append(W)
            convs.append(conv)
        
        conv = tf.concat(convs, axis=3, name='concat_dilations')
        if bias:
            b = bias_variable([n_out])
            conv = conv + b
        if norm:
            relu = tf.nn.relu(batch_norm(conv, n_out), name='relu')
        else:
            relu = tf.nn.relu(conv, name='relu')
    return weights, relu


def create_fully_connected_layer(input_, n_in, n_out=1, relu=False,
                                 name='fully_connected'):
    """create fully connected layer

    Args:
        input_ (tf.Tensor): input to convolutional layer
        n_in (int): number of dimensions in input data
        n_out (int): number of dimensions in output
        relu (bool): whether or not to include relu
        name (str): name of layer

    Returns:
        tf.Tensor, tf.Tensor: weights, outputs
    """
    with tf.variable_scope(name):
        flat = tf.reshape(input_, [-1, n_in])
        W = weight_variable([n_in, n_out])
        b = bias_variable([n_out])

        h_fc = tf.matmul(flat, W) + b
        if relu:
            h_fc = tf.nn.relu(h_fc, name='relu')
    return W, h_fc


def l2loss2rmse(l2, n):
    """convert l2 loss to rmse
    
    Args:
        l2 (float): l2 loss
        n (int): number of data points
    
    Returns:
        float: rmse
    """
    return np.sqrt(l2 * 2 / n)


class NN(object):
    """
    base class for neural networks
    """

    def __init__(self, units, save=False, learning_rate=1e-4,
                 optimizer='adam', name='', low_mem=False, num_gpus=1,
                 **kwargs):
        """
        initialize flow graph and start session

        Args:
            units (str): specifies sizes of layers
            save (bool): whether or not to save log
            learning_rate (float): learning rate for training
            optimizer (str): optimizer for training can be "gradient" for
                gradient descent, "adam", "adagrad", or "rmsprop"
            name (str): name of model for saving to file
            low_mem (bool): whether or not to use low memory mode
            num_gpu (int): number of gpus to use
        """
        self.save = save
        self.num_gpus = num_gpus
        self.name = name + '_' + \
            ''.join([random.choice(string.ascii_lowercase) for _ in range(6)])
        if optimizer not in ['descent', 'adam', 'adagrad', 'rmsprop']:
            raise ValueError('optimizer must be \'descent\', \'adagrad\', '
                             '\'rmsprop\', or \'adam\'')
        
        # inputs to model
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True))
        self.build(units, learning_rate, optimizer, **kwargs)
        self.sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=self.sess)
        if save:
            self.init_writer()
            self.save_graph()
        self.i = 0
        self.low_mem = low_mem

    def init_writer(self):
        """
        initialize writer objects
        """
        self.save = True
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(s.LOGS_DIR, self.sess.graph)

    def average_gradients(self, tower_grads, clip=True):
        """
        compute the average of gradients computed across multiple towers
        
        Args:
            tower_grads (list) : list of lists of (gradient, variable) tuples,
                the outer list is over individual gradients, the inner list is
                over the gradient calculation for each tower
            clip (bool): whether or not to clip gradients
        
        Returns:
            list: of pairs of (gradient, variable) where the gradient has been
                averaged across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # average over the 'tower' dimension.
            grad = tf.stack([g for g, _ in grad_and_vars if g is not None], 0)
            grad = tf.reduce_mean(grad, 0)

            # keep only variable from first tower
            average_grads.append((grad, grad_and_vars[0][1]))
        if clip:
            average_grads = [(tf.clip_by_norm(g, 1), v) for g, v in
                             average_grads]
        return average_grads

    def build(self, units, learning_rate, optimizer='adam', **kwargs):
        """
        build model
        
        Args:
            units (str or int): specifies number of units, either as an int, if
                only one layer, or as string 'LxU' where L is the number of
                layers and U is the number of units per layer
            learning_rate (float): specifies learning rate of optimization
            optimizer (str): method for optimization, either "gradient",
                "rmsprop", "adam" or "adagrad"
        """
        # inputs
        self.X = tf.placeholder(tf.float32, shape=[None, None, s.XDIM],
                                name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, 2], name='y')
        self.keep_prob = tf.placeholder_with_default(1., shape=None, name='keep_prob')
        
        # split dataset for each gpu
        self.Xsplit = tf.split(tf.expand_dims(self.X, 2), self.num_gpus, 0)
        self.ysplit = tf.split(self.y, self.num_gpus, 0)

        # optimizer
        if optimizer == 'descent':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise ValueError('optimizer must be \'descent\', \'adagrad\', '
                             '\'rmsprop\', or \'adam\'')
        # rnn elements
        self.parse_units(units)
        tower_grads = []
        for i in xrange(self.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope('tower', reuse=(i != 0)):
                    loss = self.build_tower(i, **kwargs)
                    grads = self.optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
        self.grads = self.average_gradients(tower_grads)
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self.yhat = tf.concat(tf.get_collection('yhats'), 0, name='all_yhat')
        self.train_step = self.optimizer.apply_gradients(self.grads)

        ratios = [tf.reshape(tf.truediv(
            gv[0] * learning_rate, gv[1]), [-1, 1])
            for gv in grads if gv[0] is not None]
        self.update_ratio = tf.concat([r for r in ratios if tf.rank(r) != 0], 0)
        
        if self.save:
            self.train_summ = tf.summary.scalar('training loss', self.loss)
            self.test_summ = tf.summary.scalar('test loss', self.loss)

    def train(self, data, epochs=10, batch_size=128, keepprob=0.5, test=None):
        """
        train NN on given data for given number of epochs

        Args:
            data (obj): object that holds data, must have get_train_batch()
                and get_data() functions that returns tuples of data, also
                must have attribute "n" that returns dataset size
            epochs (int): number of epochs to train for
            batch_size (int): number of data points per batch of training
            keepprob (float): proportion of data to keep in dropout layer
            test (obj): object similar to data that holds test data

        Returns:
            np.ndarray: loss for every 
        """
        epochlen = int(float(data.n) / batch_size / self.num_gpus)
        iters = epochs * epochlen
        period = 1000
        # get full train and test datasets if not in low memory mode
        if not self.low_mem:
            allX, ally = data.get_data()
            n = allX.shape[0]
        if test is not None:
            if self.low_mem:
                print 'evaluating test loss on a batch due to low memory mode'
            else:
                testX, testy = test.get_data()
            loss = np.zeros((2, iters/period + 1))
        else:
            loss = np.zeros((1, iters/period + 1))
        # train for given number of iterations
        for i in range(iters):
            X, y = data.get_train_batch(batch_size * self.num_gpus)
            # write losses every period
            if i % period == 0:
                if self.low_mem:
                    loss[0, i/period] = self.get_loss(X, y)
                else:
                    loss[0, i/period] = self.get_loss(X, y)
                if test:
                    if self.low_mem:
                        testX, testy = test.get_train_batch(
                            batch_size * self.num_gpus)
                    loss[1, i/period] = self.get_loss(
                        testX, testy, test=True)
                    print 'step %d: train loss %g, test loss rmse %g' % \
                        (i, loss[0, i/period], loss[1, i/period])
                else:
                    print 'step %d: loss %g' % (i, loss[0, i/period])
                sys.stdout.flush()
            # save model every 5 periods
            if i != 0 and i % (period * 5) == 0 and self.save:
                if test:
                    modelsuffix = '_%f' % loss[1, i/period]
                else:
                    modelsuffix = '_train%f' % loss[0, i/period]
                self.save_model(modelsuffix)
            self.train_batch(X, y, keepprob, i % (period * 5) == 0)
            if (i+1) % epochlen == 0:
                print '%d epochs (%d iters) finished' % ((i+1) / epochlen, i+1)
                sys.stdout.flush()
        return loss

    def train_batch(self, X, y, keepprob=0.5, update_ratio=False):
        """
        train data on single batch of data

        Args:
            X (np.ndarray): matrix representing input
            y (np.ndarray): matrix representing predictions
            keepprob (float): proportion of data to keep in dropout layer
            update_ratio (bool): whether or not to calculate update ratio
        """
        X, y = self._pad_dataset(X, y)
        if update_ratio:
            _, ur = self.sess.run([self.train_step, self.update_ratio],
                                  feed_dict={self.X: X, self.y: y,
                                             self.keep_prob: keepprob})
            ur = np.log10(ur)
            ur = ur[np.isfinite(ur)]
            print '\tupdate ratio: mean %f, median %f' % (ur.mean(),
                                                          np.median(ur))
        else:
            self.sess.run([self.train_step],
                          feed_dict={self.X: X, self.y: y,
                                     self.keep_prob: keepprob})
        self.i += 1

    def _pad_dataset(self, X, y=None):
        """ pad dataset to split evenly over GPUs

        Args:
            X (np.ndarray): matrix representing input
            y (np.ndarray): matrix representing predictions
        """
        ndata = X.shape[0]
        if ndata % self.num_gpus:
            chunksize = int(np.ceil(float(ndata)/self.num_gpus))
            X = np.pad(X, ((0, self.num_gpus * chunksize - ndata), (0, 0),
                       (0, 0)), 'constant')
            if y is not None:
                y = np.pad(y, ((0, self.num_gpus * chunksize - ndata), (0, 0)),
                           'constant')
        if y is None:
            return X
        else:
            return X, y

    def get_loss(self, X, y, test=False):
        """
        get value of loss function

        Args:
            X (np.ndarray): matrix representing input
            y (np.ndarray): matrix representing predictions
            test (bool): whether loss is for test data (for log)

        Returns:
            float: RMSE
        """
        # get variables to fetch
        if self.save:
            if test:
                fetches = [self.loss, self.test_summ]
            else:
                fetches = [self.loss, self.train_summ]
        else:
            fetches = [self.loss]

        # run operations, split into chunks if too large
        ndata = X.shape[0]
        X, y = self._pad_dataset(X, y)
        chunksize = MAXCHUNK * self.num_gpus
        if self.low_mem:
            chunksize /= 10

        nchunks = int(np.ceil(float(X.shape[0])/chunksize))
        losses = []
        for i in range(nchunks):
            j, k = i * chunksize, (i+1) * chunksize
            result = self.sess.run(fetches, feed_dict={
                self.X: X[j:k], self.y: y[j:k],
                self.keep_prob: 1})
            losses.append(result[0])

        # write results
        if self.save:
            self.writer.add_summary(result[1], self.i)

        return np.sqrt(sum(losses)/ndata)

    def save_graph(self):
        """
        save meta graph to file
        """
        tf.train.export_meta_graph('%s/%s.meta' % (s.MODELS_DIR, self.name))

    def save_model(self, suffix=''):
        """
        save variables to file

        Args:
            suffix (str): suffix for save filename
        """
        saver = tf.train.Saver()
        saver.save(self.sess, '%s/%s%s' % (s.MODELS_DIR, self.name, suffix),
                   global_step=self.i, write_meta_graph=False)

    def restore(self, filename):
        """
        restore variables from file

        Args:
            filename (str): filename to read variable values from
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        spl = filename.split('-')
        if os.path.basename(filename).startswith(self.name.split('_')[0]):
            self.name = os.path.basename(('-'.join(spl[:-1])).rsplit('_', 1)[0])
            try:
                self.i = int(spl[-1])
            except:
                self.i = 0

    def test(self, X):
        """
        test NN on given data

        Args:
            X (np.ndarray): input matrix
        
        Returns:
            np.ndarray: predicted values
        """
        ndata = X.shape[0]
        if ndata == 0:
            return np.zeros((0, 2))
        X = self._pad_dataset(X)
        chunksize = min(MAXCHUNK * self.num_gpus, X.shape[0])
        if self.low_mem and chunksize == MAXCHUNK * self.num_gpus:
            chunksize /= 10
        nchunks = int(np.ceil(float(X.shape[0])/chunksize))
        chunk_yhat = []
        for i in range(nchunks):
            j, k = i * chunksize, (i+1) * chunksize
            result = self.yhat.eval(
                session=self.sess, feed_dict={self.X: X[j:k,:], self.keep_prob: 1})
            chunk_yhat.append(result)
        return np.vstack(chunk_yhat)[0:ndata,:]


class CNN(NN):
    """
    represents a convolutional neural net
    """

    def parse_units(self, units):
        """
        parse units to list of ints

        Args:
            units (str): number of units, comma-separated
        """
        try:
            self.units = []
            for u in units.split(','):
                try:
                    self.units.append(int(u))
                except ValueError:
                    spl = u.split('x')
                    self.units.extend([int(spl[1])]*int(spl[0]))
        except:
            raise ValueError('invalid format for units, requires '
                             'comma-separated list of integers')

    def build_tower(self, i, batch_norm=False, scale=1, norm=1, dilate=False,
                    alpha=0):
        """
        build convolutional layers and loss function for CNN

        Args:
            i (int): tower number
            batch_norm (bool): whether or not to perform batch normalization
            scale (float): scale of dG relative to dH
            norm (int): order of norm for loss function
            dilate (bool): whether or not to include dilated convolutions
            alpha (float): regularization parameter for l2
        """
        self.layers = [self.Xsplit[i]]
        self.weights = []
        motif_size = tf.shape(self.Xsplit[i])[1]

        # conv layers
        for j in range(len(self.units)):
            n = s.XDIM if j == 0 else self.units[j-1]
            if dilate:
                result = create_dilated_conv_layer(
                    self.layers[j], n, self.units[j], 'conv%d' % j, j != 0,
                    batch_norm, self.keep_prob)
                self.weights.extend(result[0])
            else:
                result = create_conv_layer(
                    self.layers[j], n, self.units[j], 'conv%d' % j, j != 0,
                    batch_norm, self.keep_prob)
                self.weights.append(result[0])
            self.layers.append(result[1])
        self.layers.append(tf.concat(self.layers, 3, name='combine_layers'))
        self.shape = tf.shape(self.layers[-1])
        n = sum(self.units) + s.XDIM

        # fully connected layer
        result = create_fully_connected_layer(self.layers[-1], n, 2)
        self.weights.append(result[0])
        self.layers.append(tf.reshape(result[1], [-1, motif_size, 2]))
        
        # readout
        yhat = tf.reduce_sum(self.layers[-1], 1, name='add_energies')
        tf.add_to_collection('yhats', yhat)

        # final prediction & loss
        err = yhat - self.ysplit[i]
        if scale <= 1:
            loss = tf.norm(err[:,0], ord=norm, name='dGloss') * scale + \
                tf.norm(err[:,1], ord=norm, name='dHloss')
        else:
            loss = tf.norm(err[:,0], ord=norm, name='dGloss') + \
                tf.norm(err[:,1], ord=norm, name='dHloss') / scale
        if alpha:
            loss += alpha * tf.add_n([tf.nn.l2_loss(w) for w in self.weights[:-1]])
        tf.add_to_collection('losses', loss)
        return tf.add_n(tf.get_collection('losses', 'tower'),
                        name='tower_loss')

    def get_activation_profile(self, layer, unit, iters=1000,
                               learning_rate=1e-1):
        """
        run gradient descent over possible input matrices to maximize
        activation of specified neuron

        Args:
            layer (int): layer number
            unit (int): unit number
            iters (int): number of iterations of gradient descent
            learning_rate (float): learning rate for gradient descent
        """
        if not self.save:
            raise ValueError('cannot get unit activation with write mode off')
        X = np.random.uniform(size=[1, 8, s.XDIM]).astype('float32')
        max_activation = tf.reduce_max(tf.slice(self.layers[layer],
                                       [0, 0, 0, unit], [1, s.XDIM, 1, 1]))
        grad_X = tf.gradients(max_activation, self.X)
        for i in range(iters):
            result = self.sess.run(grad_X,
                                   feed_dict={self.X: X,
                                              self.keep_prob: 1})
            X += learning_rate*result[0]
        X_summ = tf.summary.image('activation/motif%d-%d' % (layer, unit),
                                  tf.expand_dims(self.X, 3))
        self.writer.add_summary(self.sess.run(
            X_summ, feed_dict={self.X: X, self.keep_prob: 1}))
        return

    def get_weights(self, layer):
        """
        get weight of given layer/unit

        Args:
            layer (int): layer number
        """
        return self.weights[layer].eval(self.sess)
