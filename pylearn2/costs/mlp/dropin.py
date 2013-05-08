__authors__ = 'Peter Sadowski'

from pylearn2.costs.cost import Cost

class Dropin(Cost):
    """
    This is a slight modification of Ian Goodfellow's costs/mlp/dropout.py cost.
    """

    supervised = True

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
            default_input_scale=2., input_scales=None):
        """
        During training, each input to each layer is randomly included or <set to 1>
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Should probably keep the scale == 1 for dropin algorithm.
        """

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y, ** kwargs):
        Y_hat = model.dropin_fprop(X, default_input_include_prob=self.default_input_include_prob,
                input_include_probs=self.input_include_probs, default_input_scale=self.default_input_scale,
                input_scales=self.input_scales
                )
        return model.cost(Y, Y_hat)
