"""
Multitask MLP

Modified version of Ian Goodfellow's mlp.py
"""
__authors__ = "Peter Sadowski"

from collections import OrderedDict
import numpy as np
import sys
import warnings

from theano import config
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.costs.mlp import Default
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import function
from pylearn2.utils import py_integer_types
from pylearn2.utils import sharedX

from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Layer
from pylearn2.costs.cost import SumOfCosts
import functools

class MTMLP(MLP):
    """
    A Multi-Task Multi-Layer Perceptron.
    This is a modified MLP with multiple outputs. 
    """
    
    @functools.wraps(MLP.__init__)
    def __init__(self,
            layers,
            alpha,
            batch_size=None,
            input_space=None,
            nvis=None,
            seed=None,
            dropout_include_probs = None,
            dropout_scales = None,
            dropout_input_include_prob = None,
            dropout_input_scale = None,
            ):
        """
            layers: a list of MLP_Layers. The final layer will specify the
                    MLP's output space.
            batch_size: optional. if not None, then should be a positive
                        integer. Mostly useful if one of your layers
                        involves a theano op like convolution that requires
                        a hard-coded batch size.
            input_space: a Space specifying the kind of input the MLP acts
                        on. If None, input space is specified by nvis.
            dropout*: None of these arguments are supported anymore. Use
                      pylearn2.costs.mlp.dropout.Dropout instead.
        """
        assert(alpha >= 0.0 and alpha <= 1)
        self.alpha = alpha
        #super(MTMLP, self).__init__(*args, **kwargs)
        
        locals_snapshot = locals()
        for arg in locals_snapshot:
            if arg.find('dropout') != -1 and locals_snapshot[arg] is not None:
                raise TypeError(arg+ " is no longer supported. Train using "
                        "an instance of pylearn2.costs.mlp.dropout.Dropout "
                        "instead of hardcoding the dropout into the model"
                        " itself. All dropout related arguments and this"
                        " support message may be removed on or after "
                        "October 2, 2013. They should be removed from the "
                        "SoftmaxRegression subclass at the same time.")

        if seed is None:
            seed = [2013, 1, 4]

        self.seed = seed
        self.setup_rng()

        assert isinstance(layers, list)
        assert all(isinstance(layer, Layer) for layer in layers)
        assert len(layers) >= 1
        self.layer_names = set()
        for layer in layers:
            assert layer.get_mlp() is None
            assert layer.layer_name not in self.layer_names
            layer.set_mlp(self)
            self.layer_names.add(layer.layer_name)

        self.layers = layers

        self.batch_size = batch_size
        self.force_batch_size = batch_size

        assert input_space is not None or nvis is not None
        if nvis is not None:
            input_space = VectorSpace(nvis)

        self.input_space = input_space

        # Initialize final layers. These are special for mtmlp.
        #self._update_layer_input_spaces()
        layers[0].set_input_space(self.input_space)
        for i in xrange(1,len(self.layers)-2):
            layers[i].set_input_space(layers[i-1].get_output_space())
        layers[-2].set_input_space(layers[-3].get_output_space())
        layers[-1].set_input_space(layers[-3].get_output_space())

        self.freeze_set = set([])

        def f(x):
            if x is None:
                return None
            return 1. / x

    #def get_output_space(self):
    #    return self.layers[-1].get_output_space()
    
    def get_monitoring_channels(self, X=None, Y=None):
        """
        Note: X and Y may both be None, in the case when this is
              a layer of a bigger MLP.
        """

        states = self.fprop(X, return_all=True)
        rval = OrderedDict()

        for layer,state in zip(self.layers, states):
            ch = layer.get_monitoring_channels()
            for key in ch:
                rval[layer.layer_name+'_'+key] = ch[key]
            args = [state]
            if layer is self.layers[-1]:
                args.append(Y)
            ch = layer.get_monitoring_channels_from_state(*args)
            if not isinstance(ch, OrderedDict):
                raise TypeError(str((type(ch), layer.layer_name)))
            for key in ch:
                rval[layer.layer_name+'_'+key]  = ch[key]

        return rval

    @functools.wraps(MLP.fprop)
    def fprop(self, state_below, return_all = False):
        # Returns tuple (output layer -2, output layer -1)
        rval = state_below
        rlist = []

        # Propogate through all but last two layers.
        for i, layer in enumerate(self.layers[:-2]):
            # Standard fprop.
            rval = layer.fprop(rval)
            rlist.append(rval)
        
        # Last layer is connected to 2nd layer down.
        rlist.append(self.layers[-2].fprop(rval))
        rlist.append(self.layers[-1].fprop(rval))
            
        if return_all:
            # Return full list of layer activations
            return rlist
        # Return tuple of output layer activations.
        return (rlist[-2], rlist[-1])

    def cost_supervised(self, Y, Y_hat):
        return self.layers[-1].cost(Y, Y_hat)

    def cost_unsupervised(self, X, X_hat):
        return self.layers[-2].cost(X, X_hat)

    @functools.wraps(MLP.cost_matrix)
    def cost_matrix(self, Y, Y_hat):
        raise NotImplementedError(str(type(self))+" does not implement cost_matrix.")
        #return self.layers[-1].cost_matrix(Y, Y_hat)

    @functools.wraps(MLP.cost_from_cost_matrix)
    def cost_from_cost_matrix(self, cost_matrix):
        #return self.layers[-1].cost_from_cost_matrix(cost_matrix)
        raise NotImplementedError(str(type(self))+" does not implement cost_from_cost_matrix.")

    @functools.wraps(MLP.cost_from_X)
    def cost_from_X(self, X, Y):
        """
        Computes self.cost, but takes X rather than Y_hat as an argument.
        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)
        """
        X_hat, Y_hat = self.fprop(X)        
        unsupervisedcost = self.cost_unsupervised(X, X_hat)
        supervisedcost   = self.cost_supervised(Y, Y_hat)
        # TODO: check that unsupervisedcost is comparable to supervised.
        return self.alpha*supervisedcost + (1.0-self.alpha)*unsupervisedcost

