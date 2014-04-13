__authors__ = 'Peter Sadowski'
'''
Additive Gaussian noise for each unit's activity.
'''

from pylearn2.costs.cost import Cost

class AddGauss(Cost):
    """
    This is a slight modification of Ian Goodfellow's costs/mlp/dropout.py cost.
    """

    supervised = True

    def __init__(self, input_noise_stdev, input_scales=None):
        """
        During training, an independent random variable delta_i is 
        added to each input to each layer for each example. 
         
        """

        assert input_scales is None, 'Scaled input not implemented.'
        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y, ** kwargs):
        Y_hat = model.addgauss_fprop(X, input_noise_stdev=self.input_noise_stdev)
        return model.cost(Y, Y_hat)
