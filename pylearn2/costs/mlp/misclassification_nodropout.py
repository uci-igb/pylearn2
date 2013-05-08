# Cost for 
__author__ = 'Peter Sadowski'


import theano.tensor as T
from theano import config
from pylearn2.costs.cost import Cost
import numpy as np

class Misclassification_nodropout(Cost):
    """
    The misclassification rate, using the final mlp without dropout.
    """
    supervised = True

    def __call__(self, model, X, Y):
        y_hat = model.fprop(X)
        y_hat = T.argmax(y_hat, axis=1)
        y = T.argmax(Y, axis=1)
        misclass = T.neq(y, y_hat).mean()
        misclass = T.cast(misclass, config.floatX)
        return misclass
