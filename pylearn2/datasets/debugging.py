__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as N
np = N
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.utils import serial
from pylearn2.utils.mnist_ubyte import read_mnist_images
from pylearn2.utils.mnist_ubyte import read_mnist_labels

import pickle as pkl
import os

class randombinary(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set,
                       nexamples=1000, 
                       nfeatures=100, 
                       p=0.5, 
                       nimportant=1 
                       ):
        self.args = locals()
       
        path = os.environ['PYLEARN2_DATA_PATH'] + "/debugging/"
        filename = path + "nex%d_nfeat%d_p%d_nimp%d.pkl" % (nexamples, nfeatures, p*100, nimportant)
        try:
            f = open(filename, 'r')
            datadict = pkl.load(f)
            f.close()
            print('Loaded data from %s' % filename)
        except IOError:
            print('\nCreating dataset and saving to %s' % filename)
            datadict = self.makedataset(nexamples=nexamples, nfeatures=nfeatures, p=p, nimportant=nimportant)
            f = open(filename, 'w')
            pkl.dump(datadict, f)
            f.close()
        
        X_train = datadict['X_train']
        Y_train = datadict['Y_train']
        X_test  = datadict['X_test']
        Y_test  = datadict['Y_test']
        assert(X_train.shape == (nexamples, nfeatures))
        assert(Y_train.shape == (nexamples, 1))
        assert(X_test.shape == (nexamples, nfeatures))
        assert(Y_test.shape == (nexamples, 1))

        if which_set == 'train':
            super(randombinary,self).__init__(X=X_train, y=Y_train)
            self.X = X_train
            self.y = Y_train
        elif which_set == 'test':
            super(randombinary,self).__init__(X=X_test, y=Y_test)
            self.X = X_test
            self.y = Y_test
        else:
            raise ValueError('Unrecognized which_set value "%s".' %
                    (which_set,)+'". Valid values are ["train","test"].')

    def get_test_set(self):
        return randombinary('test') 

    def makedataset(self, nexamples, nfeatures, p, nimportant):
        ''' Make a random dataset.'''
        assert(p >= 0)
        assert(p <= 1)
        assert(nexamples > 0)
        assert(nfeatures > 0)
        assert(nimportant >= 1)
        # Initialize data matrix randomly.
        X = np.random.binomial(1, p, (2*nexamples, nfeatures))
        # The first half of the examples will be class 1.
        # In class 1, the first nimportant features will be the same.
        mask = np.asarray(np.random.binomial(1, p, (1, nimportant)), dtype='float32')
        for i in range(X.shape[0]/2):
            X[i, 0:nimportant] = mask
        for i in range(X.shape[0]/2, X.shape[0]):
            if np.all(X[i, 0:nimportant] == mask):
                X[i, 0] = not mask[0,0]
        Y = np.all(X[:, 0:nimportant] == mask, axis=1)
        Y = np.asarray(Y, dtype='float32')
        X = np.asarray(X, dtype='float32')
        assert not N.any(N.isnan(X))
        assert not N.any(N.isnan(Y))

        # Permute
        perm = np.random.permutation(X.shape[0])
        X = X[perm,:]
        Y = Y[perm,:]
       
        # Theano expects 2d arrays.
        Y = np.atleast_2d(Y).T

        # Split into train, test.
        data = {}
        data['X_train'] = X[:nexamples, :]
        data['Y_train'] = Y[:nexamples, :]
        data['X_test']  = X[nexamples:, :]
        data['Y_test']  = Y[nexamples:, :]
       
        return data


    
class MNIST_rotated_background(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, center = False, one_hot = False):
        path = "${PYLEARN2_DATA_PATH}/mnist/mnist_rotation_back_image/"+which_set

        obj = serial.load(path)
        X = obj['data']
        X = N.cast['float32'](X)
        y = N.asarray(obj['labels'])

        self.one_hot = one_hot
        if one_hot:
            one_hot = N.zeros((y.shape[0],10),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot

        if center:
            X -= X.mean(axis=0)

        view_converter = dense_design_matrix.DefaultViewConverter((28,28,1))

        super(MNIST,self).__init__(X = X, y = y, view_converter = view_converter)

        assert not N.any(N.isnan(self.X))

