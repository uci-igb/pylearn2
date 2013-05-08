''' 
Extension for calculating statistics of unit activation during dropout training.
Author: Peter Sadowski
'''

import numpy
np = numpy
from pylearn2.train_extensions import TrainExtension
import theano
import theano.tensor as T
from pylearn2.utils import serial

import pdb

class Divergence(TrainExtension):
    """
    A callback which keeps track of a model's best parameters based on its
    performance for a given cost on a given dataset.
    """
    def __init__(self, model, monitoring_dataset, input_include_probs, input_scales):
        """
        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model whose best parameters we want to keep track of
        cost : tensor_like
            cost function used to evaluate the model's performance
        monitoring_dataset : pylearn2.datasets.dataset.Dataset
            dataset on which to compute the cost
        batch_size : int
            size of the batches used to compute the cost
        """
        self.model = model
        self.dataset = monitoring_dataset
        self.minibatch = T.matrix('minibatch')
        # Function for propagating with dropout.
        self.single_fprop_dropout = theano.function(inputs=[self.minibatch],
                                        outputs=model.dropout_fprop(self.minibatch, 
                                        input_include_probs=input_include_probs,
                                        input_scales=input_scales,
                                        return_all=True))
        # Function for propagating without dropout.
        self.single_fprop_nodropout = theano.function(inputs=[self.minibatch],
                                        outputs=model.fprop(self.minibatch, 
                                        return_all=True))

    def on_monitor(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier. If it's the
        case, records the model's parameters.
        Parameters
        ----------
        model : pylearn2.models.model.Model
            not used
        dataset : pylearn2.datasets.dataset.Dataset
            not used
        algorithm : TrainingAlgorithm
            not used
        """
        #args aren't used, because we want to do fast computations, but we accept for convenience.
        it = self.dataset.iterator('sequential',
                                    num_batches=1,
                                    targets=False)
    
        input = it.next()
        a = self.single_fprop_dropout(input)
        #activities = self.model.dropout_fprop_analyze(input) # input is state_below
        
        print('Layer 0 activity stats: %d %d %f %f %f' % (a[0].shape[0], a[0].shape[1], a[0].mean(), a[0].std(), a[0].min())) 

            #self.best_params = self.model.get_param_values()

    def get_best_params(self):
        """
        Returns the best parameters up to now for the model.
        """
        return 0

    def single_fprop(self, inputs=None, batch_size=None, dropout=False):
        """
        Return activations of all neurons for a single 
        FF propagation through network with dropout.
        Dropped neuron activations are given before dropout applied.

        inputs = numpy data array
        batch_size = number of examples to run through network
        dropout = whether or not to perform dropout (parameters specified in init)
        """
        if inputs is None:
            if batch_size is None:
                batch_size = 1
            it = self.dataset.iterator('sequential', batch_size=batch_size,targets=False)
            inputs = it.next()
        else:
            assert(batch_size is None)

        if dropout:
            a = self.single_fprop_dropout(inputs)
        else:
            a = self.single_fprop_nodropout(inputs)
        return a

class SaveAtIntervals(TrainExtension):
    """
    A callback that saves a copy of the model at regular intervals.
    """
    def __init__(self, save_prefix, interval):
        """
        Parameters
        ----------
        channel_name: the name of the channel we want to minimize.
        save_path: the path to save the best model to.
        interval: Period of saves, in epochs.
        """
        #self.__dict__.update(locals())
        self.channel_name = 'train_objective'
        self.save_prefix = save_prefix
        self.interval  = interval

    def on_monitor(self, model, dataset, algorithm):
        """
        Save the model if we are on a save epoch.
        
        Parameters
        ----------
        model : pylearn2.models.model.Model
                model.monitor must contain a channel with name given by self.channel_name
        dataset : pylearn2.datasets.dataset.Dataset
            not used
        algorithm : TrainingAlgorithm
            not used
        """

        #monitor = model.monitor
        #channels = monitor.channels
        #channel = channels[self.channel_name]
        #val_record = channel.val_record
        #epoch = len(val_record)
        epoch = model.monitor.get_epochs_seen()

        save_file = '%s_%d.pkl' % (self.save_prefix, epoch)

        if np.mod(epoch, self.interval) == 0:
            print('Saving model to %s' % save_file)
            serial.save(save_file, model, on_overwrite = 'backup')
        
