"""
The module defining the Monitor and MonitorChannel objects used for
tracking the changes in values of various quantities throughout training
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from collections import OrderedDict
import copy
import time
import warnings

from theano.printing import var_descriptor
import theano.sparse

from pylearn2.config import yaml_parse
from pylearn2.datasets.dataset import Dataset
from pylearn2.utils import function
from pylearn2.utils.string_utils import number_aware_alphabetical_key
from pylearn2.utils import sharedX
from theano import config
import numpy as np
from theano import tensor as T
from pylearn2.utils import safe_izip
from pylearn2.utils.timing import log_timing
import logging

log = logging.getLogger(__name__)


class Monitor(object):
    """
    A class for monitoring Models while they are being trained.

    A monitor object records the number of minibatches and number of examples
    the model has trained, as well as any number of "channels" that track
    quantities of interest (examples: the objective function, measures of
    hidden unit activity, reconstruction error, sum of squared second
    derivatives, average norm of the weight vectors,  etc.)
    """
    def __init__(self, model):
        """
        Makes a monitor for `model`. Assumes the model has not been
        trained at all yet.

        Parameters
        ----------
        model : pylearn2.models.model.Model instance
        """
        self.training_succeeded = False
        self.model = model
        self.channels = OrderedDict()
        self._num_batches_seen = 0
        self._examples_seen = 0
        self._epochs_seen = 0
        self._datasets = []
        self._iteration_mode = []
        self._batch_size = []
        self._num_batches = []
        self._dirty = True
        self._rng_seed = []
        self.names_to_del = ['theano_function_mode']
        self.t0 = time.time()
        # Determine whether the model should use topological or vector form of
        # examples. If the model acts on a space with more than the batch index
        # and channel dimension, the model has topological dimensions, so the
        # topological view of the data should be used.
        vector = model.get_input_space().make_theano_batch()
        if isinstance(vector.type, theano.sparse.SparseType):
            self.topo = False
        else:
            self.topo = len(vector.type.broadcastable) > 2

        self.require_label = False
        self.theano_function_mode = None

    def set_theano_function_mode(self, mode):
        if self.theano_function_mode != mode:
            self._dirty = True
            self.theano_function_mode = mode

    def add_dataset(self, dataset, mode='sequential', batch_size=None,
                    num_batches=None, seed = None):
        """
        Determines the data used to calculate the values of each channel.

        Parameters
        ----------
        dataset : object
            A `pylearn2.datasets.Dataset` object.
        mode : str or object, optional
            Iteration mode; see the docstring of the `iterator` method
            on `pylearn2.datasets.Dataset` for details.
        batch_size : int, optional
            The size of an individual batch. Optional if `mode` is
            'sequential' and `num_batches` is specified (batch size
            will be calculated based on full dataset size).
        num_batches : int, optional
            The total number of batches. Unnecessary if `mode` is
            'sequential' and `batch_size` is specified (number of
            batches will be calculated based on full dataset size).
        """
        # The user can ommit using lists if only one dataset is set
        if not isinstance(dataset, list):
            dataset = [dataset]
        if not isinstance(mode, list):
            mode = [mode]
        if not isinstance(batch_size, list):
            batch_size = [batch_size]
        if not isinstance(num_batches, list):
            num_batches = [num_batches]
        if seed is None:
            seed = [ None ] * len(dataset)
        if not isinstance(seed, list):
            seed = [ seed ]
        if any([len(l) != len(dataset) for l in [mode, batch_size, seed]]):
            raise ValueError("make sure each dataset has its iteration " + \
                        "mode, batch size and number of batches.")
        for (d, m, b, n, sd) in safe_izip(dataset, mode, batch_size, num_batches, seed):
            try:
                it = d.iterator(mode=m, batch_size=b,
                                      num_batches=n,
                                      topo=self.topo,
                                      targets=self.require_label,
                                      rng = sd)
            except ValueError as exc:
                raise ValueError("invalid iteration parameters in "
                                 "Monitor.add_dataset: " + str(exc))
            if it.stochastic:
                # must be a seed, not a random number generator
                # if it were a random number generator, different iterators using
                # it would update its state, so we would not get the same iterator
                # each time
                # Also, must not be None, because this makes the iterator pick
                # a seed based on the clock
                if not isinstance(sd, (list, tuple, int)):
                    raise TypeError("Monitor requires a seed (not a random number generator) when using stochastic iteration modes.")
            else:
                assert sd is None # the iterator should catch this, but let's double-check

            if not d in self._datasets:
                self._datasets.append(d)
                self._iteration_mode.append(m)
                self._batch_size.append(b)
                self._num_batches.append(n)
                self._rng_seed.append(sd)

    def __call__(self):
        """
        Runs the model on the monitoring dataset in order to add one
        data point to each of the channels.
        """

        # If the channels have changed at all, we need to recompile the theano
        # functions used to compute them
        if self._dirty:
            self.redo_theano()

        model = self.model
        datasets = self._datasets

        # Set all channels' val_shared to 0
        self.begin_record_entry()

        for d, i, b, n, a, sd, ne in safe_izip(datasets, self._iteration_mode, self._batch_size,
                                 self._num_batches, self.accum, self._rng_seed, self.num_examples):
            if isinstance(d, basestring):
                d = yaml_parse.load(d)
                raise NotImplementedError()
                # need to put d back into self._datasets
            myiterator = d.iterator(mode=i,
                                    batch_size=b,
                                    num_batches=n,
                                    topo=self.topo,
                                    targets=self.require_label,
                                    rng=sd)

            actual_ne = 0
            for X in myiterator:
                if self.require_label:
                    X, y = X
                    self.run_prereqs(X,y,d)
                    a(X, y)
                else:
                    self.run_prereqs(X, None, d)
                    a(X)
                if X.ndim == 2:
                    actual_ne += X.shape[0]
                else:
                    actual_ne += X.shape[d.get_topo_batch_axis()]
            # end for X
            if actual_ne != ne:
                raise RuntimeError("At compile time, your iterator said it had "
                        + str(ne) + " examples total, but at runtime it gave us "
                        + str(actual_ne) + ".")
        # end for d


        log.info("Monitoring step:")
        log.info("\tEpochs seen: %d" % self._epochs_seen)
        log.info("\tBatches seen: %d" % self._num_batches_seen)
        log.info("\tExamples seen: %d" % self._examples_seen)
        t = time.time() - self.t0
        for channel_name in sorted(self.channels.keys(), key=number_aware_alphabetical_key):
            channel = self.channels[channel_name]
            channel.time_record.append(t)
            channel.batch_record.append(self._num_batches_seen)
            channel.example_record.append(self._examples_seen)
            channel.epoch_record.append(self._epochs_seen)
            val = channel.val_shared.get_value()
            channel.val_record.append(val)
            # TODO: use logging infrastructure so that user can configure
            # formatting
            if abs(val) < 1e4:
                val_str = str(val)
            else:
                val_str = '%.3e' % val

            log.info("\t%s: %s" % (channel_name, val_str))

    def run_prereqs(self, X, y, dataset):
        if dataset not in self.prereqs:
            return
        for prereq in self.prereqs[dataset]:
            prereq(X,y)

    def get_batches_seen(self):
        """ Returns the number of batches the model has learned on (assuming
        that the learning code has been calling Monitor.report_batch correctly)
        """
        return self._num_batches_seen

    def get_epochs_seen(self):
        return self._epochs_seen

    def get_examples_seen(self):
        """ Returns the number of examples the model has learned on (assuming
        that the learning code has been calling Monitor.report_batch correctly)
        """
        return self._examples_seen

    def report_batch(self, num_examples):
        """ Call this whenever the model has learned on another batch of examples.
        Report how many examples were learned on. """
        self._examples_seen += num_examples
        self._num_batches_seen += 1

    def report_epoch(self):
        self._epochs_seen += 1

    def redo_theano(self):
        """
        Recompiles Theano functions used by this monitor.

        This is needed so that if new channels are added, Theano's
        optimizations make sure (to the extent that they can) that the new
        channels and old channels don't have any redundant calculations.

        It is also needed to regenerate Theano functions after pickling and
        unpickling, since Theano functions should not be pickled.
        """
        self._dirty = False

        init_names = dir(self)
        self.prereqs = OrderedDict()
        for channel in self.channels.values():
            if channel.prereqs is not None:
                dataset = channel.dataset
                if dataset not in self.prereqs:
                    self.prereqs[dataset] = []
                prereqs = self.prereqs[dataset]
                for prereq in channel.prereqs:
                    if prereq not in prereqs:
                        prereqs.append(prereq)

        updates = OrderedDict()
        for channel in self.channels.values():
            updates[channel.val_shared] = np.cast[config.floatX](0.0)
        with log_timing(log, "compiling begin_record_entry"):
            self.begin_record_entry = function(inputs=[], updates=updates, mode=self.theano_function_mode,
                    name = 'Monitor.begin_record_entry')
        updates = OrderedDict()
        givens = OrderedDict()
        # Get the appropriate kind of theano variable to represent the data the model
        # acts on
        X = self.model.get_input_space().make_theano_batch(name = "monitoring_X")
        if config.compute_test_value != 'off':
            m = self.model.get_test_batch_size()
            test_value = self.model.get_input_space().get_origin_batch(m)
            X.tag.test_value = np.cast[X.type.dtype](test_value)
        if self.require_label:
            Y = self.model.get_output_space().make_theano_batch(name = "monitoring_Y")

        log.info('Monitored channels: ')
        for key in sorted(self.channels.keys()):
            mode = self.theano_function_mode
            if mode is not None and hasattr(mode, 'record'):
                mode.record.handle_line('compiling monitor including channel '+key+'\n')
            log.info('\t%s' % key)
        it = [d.iterator(mode=i, num_batches=n, batch_size=b, topo=self.topo) \
              for d, i, n, b in safe_izip(self._datasets, self._iteration_mode,
                                    self._num_batches, self._batch_size)]
        self.num_examples = [np.cast[config.floatX](float(i.num_examples)) for i in it]
        givens = [OrderedDict() for d in self._datasets]
        updates = [OrderedDict() for d in self._datasets]
        for channel in self.channels.values():
            index = self._datasets.index(channel.dataset)
            d = self._datasets[index]
            g = givens[index]
            cur_num_examples = self.num_examples[index]
            u = updates[index]
            if isinstance(channel.graph_input, (list, tuple)):
                channel_X, channel_Y = channel.graph_input
                assert channel_X not in g or g[channel_X] is X
                assert channel_Y not in g or g[channel_Y] is Y
                g[channel_X] = X
                g[channel_Y] = Y
            else:
                channel_X = channel.graph_input
                assert channel_X not in g or g[channel_X] is X
                g[channel_X] = X
            if n == 0:
                raise ValueError("Iterating over 0 examples results in divide by 0")
            if self.topo:
                batch_index = d.get_topo_batch_axis()
            else:
                batch_index = 0
            val = channel.val * T.cast(X.shape[batch_index], config.floatX) / cur_num_examples
            u[channel.val_shared] = channel.val_shared + val

        with log_timing(log, "Compiling accum"):
            # Check type of update expressions
            for up in updates:
                for key in up:
                    if key.dtype != up[key].dtype:
                        raise TypeError('Monitoring channel shared variable ' \
                                + key.name + ' has dtype ' + key.dtype + \
                                ' but is driven by an expression with type ' + \
                                up[key].dtype)

            self.accum = []
            for idx, packed in enumerate(safe_izip(givens, updates)):
                g, u = packed
                mode = self.theano_function_mode
                if mode is not None and hasattr(mode, 'record'):
                    for elem in g:
                        mode.record.handle_line('g key '+var_descriptor(elem)+'\n')
                        mode.record.handle_line('g val '+var_descriptor(g[elem])+'\n')
                    for elem in u:
                        mode.record.handle_line('u key '+var_descriptor(elem)+'\n')
                        mode.record.handle_line('u val '+var_descriptor(u[elem])+'\n')
                function_name = 'Monitor.accum[%d]' % idx
                if self.require_label:
                    if mode is not None and hasattr(mode, 'record'):
                        mode.record.handle_line('compiling supervised accum\n')
                    # Some channels may not depend on the data, ie, they might just monitor the model
                    # parameters, or some shared variable updated by the training algorithm, so we
                    # need to ignore the unused input error
                    self.accum.append(function([X, Y], givens=g, updates=u, mode=self.theano_function_mode,
                            name=function_name))
                else:
                    if mode is not None and hasattr(mode, 'record'):
                        mode.record.handle_line('compiling unsupervised accum\n')
                    self.accum.append(function([X], givens=g, updates=u, mode=self.theano_function_mode,
                            name=function_name))
            for a in self.accum:
                if mode is not None and hasattr(mode, 'record'):
                    for elem in a.maker.fgraph.outputs:
                        mode.record.handle_line('accum output '+var_descriptor(elem)+'\n')
                log.info("graph size: %d" % len(a.maker.fgraph.toposort()))
        final_names = dir(self)
        self.register_names_to_del([name for name in final_names
                                    if name not in init_names])

    def register_names_to_del(self, names):
        """
        Register names of fields that should be deleted before pickling.

        Parameters
        ----------
        names : list
            A list of attribute names as strings.
        """
        for name in names:
            if name not in self.names_to_del:
                self.names_to_del.append(name)

    def __getstate__(self):
        """
        In order to avoid pickling a copy of the dataset whenever a monitor
        is saved, the __getstate__ method replaces the dataset field with the
        dataset's yaml source. This is not a perfect solution because it won't
        work with job resuming, which would require saving the state of the
        dataset's random number generator.

        Like in the Model class, we also need to avoid saving any Theano
        functions, so we delete everything that can be regenerated with
        `redo_theano` by deleting the fields in `self.names_to_del`
        """

        # Patch old pickled monitors
        if not hasattr(self, '_datasets'):
            self._datasets = [ self._dataset ]
            del self._dataset

        temp = self._datasets

        if self._datasets:
            self._datasets = []
            for dataset in temp:
                if isinstance(dataset, basestring):
                    self._datasets.append(dataset)
                else:
                    try:
                        self._datasets.append(dataset.yaml_src)
                    except AttributeError:
                        warnings.warn('Trained model saved without indicating yaml_src')
        d = copy.copy(self.__dict__)
        self._datasets = temp
        for name in self.names_to_del:
            if name in d:
                del d[name]


        return d

    def __setstate__(self, d):

        # patch old pkl files
        if '_dataset' in d:
            d['_datasets'] = [ d['_dataset'] ]
            del d['_dataset']

        self.__dict__.update(d)

    def add_channel(self, name, ipt, val, dataset=None, prereqs=None):
        """
        Asks the monitor to start tracking a new value.  Can be called even
        after the monitor is already in use.

        Parameters
        ----------
        name: str
            The display name in the monitor.
        ipt: tensor_like
            The symbolic tensor which should be clamped to the data.
            (or a (features,targets) list/tuple containing two symbolic tensors)
        val: tensor_like
            The value (function of `ipt`) to be tracked.
        dataset: A Dataset instance specifying which dataset to compute
            this channel on.
        prereqs: list of callables that take two numpy tensors
            (X and y, where y will be None if no labels are used)
            each prereq must be called exactly once per each new
            batch of data drawn *from dataset* before the channel
            value is computed
            if two channels provide a prereq with exactly the same
            id, that prereq will only be called once
        """

        if isinstance(val, (float, int, long)):
            val = np.cast[theano.config.floatX](val)

        val = T.as_tensor_variable(val)

        if not isinstance(ipt, (list, tuple)):
            tmp = [ ipt ]
        else:
            tmp = ipt
        inputs = theano.gof.graph.inputs([val])
        for elem in inputs:
            if not hasattr(elem, 'get_value') and not isinstance(elem, theano.gof.graph.Constant):
                if elem not in tmp:
                    raise ValueError("Unspecified input: "+str(elem))



        mode = self.theano_function_mode
        if mode is not None and hasattr(mode, 'record'):
            mode.record.handle_line('Adding monitor channel '+name+'\n')
            if isinstance(ipt, (list, tuple)):
                for elem in ipt:
                    mode.record.handle_line('Includes input var '+var_descriptor(elem)+'\n')
            else:
                mode.record.handle_line(name+' input var is '+var_descriptor(ipt)+'\n')
            mode.record.handle_line('channel '+name+' is '+var_descriptor(val)+'\n')

        if dataset is None:
            if len(self._datasets) == 1:
                dataset = self._datasets[0]
            elif len(self._datasets) == 0:
                raise ValueError(_err_no_data)
            else:
                raise ValueError(_err_ambig_data)

        try:
            self._datasets.index(dataset)
        except ValueError:
            raise ValueError("The dataset specified is not " + \
                "one of the monitor's datasets")

        if name in self.channels:
            raise ValueError("Tried to create the same channel twice (%s)" %
                             name)
        if isinstance(ipt, (list, tuple)):
            if dataset is not None:
                if not dataset.has_targets():
                    raise ValueError("Tried to create a channel ("+name \
                            +") that uses targets, but monitoring dataset has no targets")
            self.require_label = True
            assert len(ipt) == 2
        self.channels[name] = MonitorChannel(ipt, val, name, dataset, prereqs)
        self._dirty = True

    def _sanity_check(self):
        """
            Sometimes we serialize models and then load them somewhere else
            but still try to use their Monitor, and the Monitor is in a mangled
            state. I've added some calls to _sanity_check to try to catch when
            that happens. Not sure what to do for a long term fix. I think it
            requires making theano graphs serializable first.
        """
        for name in self.channels:
            channel = self.channels[name]
            assert hasattr(channel, 'prereqs')

    @classmethod
    def get_monitor(cls, model):
        """
        Returns a model's monitor. If the model doesn't have a monitor yet,
        installs one and returns that.

        Parameters
        ----------
        model : object
            An object that implements the `Model` interface specified in
            `pylearn2.models`.
        """

        if hasattr(model, 'monitor'):
            rval = model.monitor
            rval._sanity_check()
        else:
            rval = Monitor(model)
            model.monitor = rval

        return rval

    # TODO: find out if monitor.foo below are used anywhere, remove if not.
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batches(self):
        return self._num_batches

    def setup(self, dataset, cost, batch_size, num_batches = None, extra_costs=None):
        """
        Sets up the monitor for a cost minimization problem.
        Adds channels defined by both the model and the cost for
        the specified dataset(s), as well as a channel called 'objective'
        defined by the costs' __call__ method.

        dataset: a Dataset or dictionary mapping string names to Datasets
                    If string names are used, then for every dataset,
                    each channel defined by the model or cost will be
                    replicated with that dataset's name followed by an
                    underscore as the prefix.
                    For example, if your cost defines a channel called
                    'misclass', and datasets is {'train' : train_dataset,
                    'valid' : valid_dataset} you will get channels called
                    'train_misclass' and 'valid_misclass'.

        cost: a Cost

        """
        if dataset is None:
            return
        if isinstance(dataset, Dataset):
            dataset = {'': dataset}
        else:
            assert isinstance(dataset, dict)
            assert all(isinstance(key, str) for key in dataset)
            assert all(isinstance(dataset[key], Dataset) for key in dataset)

        if extra_costs is None:
            costs = {}
        else:
            costs = extra_costs
        assert '' not in costs
        costs[''] = cost

        supervised = any(cost.supervised for cost in costs.values())
        model = self.model

        X = model.get_input_space().make_theano_batch()
        X.name = 'monitor_X'

        if supervised:
            Y = model.get_output_space().make_theano_batch()
            Y.name = 'monitor_Y'
            ipt = (X, Y)
        else:
            Y = None
            ipt = X
        custom_channels = {}
        for cost_name in costs:
            if cost_name == '':
                prefix = ''
            else:
                prefix = cost_name + '_'
            cost = costs[cost_name]
            raw_channels = cost.get_monitoring_channels(model, X, Y)
            channels = {}
            for name in raw_channels:
                channels[prefix+name] = raw_channels[name]
            custom_channels.update(channels)
        model_channels = model.get_monitoring_channels(X, Y)
        custom_channels.update(model_channels)
        for dataset_name in dataset:
            cur_dataset = dataset[dataset_name]
            self.add_dataset(dataset=cur_dataset,
                                 mode='sequential',
                                 batch_size=batch_size,
                                 num_batches=num_batches)
            if dataset_name == '':
                dprefix = ''
            else:
                dprefix = dataset_name + '_'
            # These channel name 'objective' must not vary, since callbacks that respond to the
            # values in the monitor use the name to find it.
            for cost_name in costs:
                cost = costs[cost_name]
                cost_value = cost(model, X, Y)
                if cost_value is not None:
                    if cost_name == '':
                        name = dprefix + 'objective'
                    else:
                        name = dprefix + cost_name
                    self.add_channel(name=name, ipt=ipt,
                        val=cost_value, dataset=cur_dataset)
            for key in custom_channels:
                self.add_channel(name=dprefix + key, ipt=ipt,
                        val=custom_channels[key], dataset=cur_dataset)


class MonitorChannel(object):
    """
    A class representing a specific quantity to be monitored.
    """
    def __init__(self, graph_input, val, name, dataset, prereqs=None):
        """
        Creates a channel for a quantity to be monitored.

        Parameters
        ----------
        graph_input : tensor_like
            The symbolic tensor which should be clamped to the data.
        val : tensor_like
            The value (symbolic function of `graph_input`) to be evaluated
            and recorded.
        name : str
            The display name in the monitor.
        prereqs: list of callables that take numpy tensors
            each prereq must be called exactly once per each new
            batch of data before the channel value is computed
            if two channels provide a prereq with exactly the same
            id, that prereq will only be called once
        """
        self.name = name
        self.prereqs = prereqs
        self.graph_input = graph_input
        if isinstance(val, float):
            val = T.constant(np.cast[config.floatX](val))
        self.val = val
        self.val_shared = sharedX(0.0, name + "_tracker")
        assert self.val_shared.dtype == config.floatX
        if not hasattr(val,'dtype'):
            raise TypeError('Monitor channel '+name+' has value of type '+str(type(val)))
        if val.dtype != self.val_shared.dtype:
            raise ValueError('monitor channels are expected to have dtype ' \
                    +str(self.val_shared.dtype) + ' but "'+name+'" has dtype '\
                    +str(val.dtype))
        if val.ndim != 0:
            raise ValueError('monitor channels are supposed to have zero dimensions ' \
                    ' but "'+name+'" has '+str(val.ndim))
        # Dataset monitored by this channel
        self.dataset = dataset
        # Value of the desired quantity at measurement time.
        self.val_record = []
        # Number of batches seen at measurement time.
        self.batch_record = []
        # Number of examples seen at measurement time (batch sizes may
        # fluctuate).
        self.example_record = []
        self.epoch_record = []
        self.time_record = []

    def __str__(self):
        try:
            graph_input_str = str(self.graph_input)
        except:
            graph_input_str = '<bad graph input>'

        try:
            val_str = str(self.val)
        except:
            val_str = '<bad val>'

        try:
            name_str = str(self.name)
        except:
            name_str = '<bad name>'

        try:
            prereqs_str = str(self.prereqs)
        except:
            prereqs_str = '<bad prereqs>'

        return "MonitorChannel(%s,%s,%s,%s)" % (graph_input_str,
                val_str,
                name_str,
                prereqs_str)

    def __getstate__(self):
        """ TODO:
                we need to figure out a good way of saving the other fields.
                in the current setup, since there's no good way of coordinating
                with the model/training algorithm, the theano based fields might
                be invalid after a repickle.
                This means we can't, for instance, resume a job with monitoring
                after a crash.
                For now, to make sure no one erroneously depends on these bad
                values, I exclude them from the pickle.
        """
        return {
            'example_record': self.example_record,
            'batch_record' : self.batch_record,
            'time_record' : self.time_record,
            'epoch_record' : self.epoch_record,
            'val_record': self.val_record
        }

    def __setstate__(self, d):
        self.__dict__.update(d)
        if 'batch_record' not in d:
            self.batch_record = [None] * len(self.val_record)
        # Patch old pickle files that don't have the "epoch_record" field
        if 'epoch_record' not in d:
            # This is not necessarily correct but it is in the most common use
            # case where you don't add monitoring channels over time.
            self.epoch_record = range(len(self.val_record))
        if 'time_record' not in d:
            self.time_record = [ None ] * len(self.val_record)


def push_monitor(model, name, transfer_experience = False):
    """
    When you load a model in a yaml file and you want to store its
    old monitor under a different name and start a new monitor, wrap
    the model in this function call.

    model: The model you loaded
    name: Will save the old monitor to model.name
    transfer_experience: If True, the new monitor will start with its
        epochs seen, batches seen, and examples seen set to where the
        old monitor left off. This is nice for stitching together learning
        curves across multiple stages of learning.
    """

    assert hasattr(model, 'monitor')
    old_monitor = model.monitor
    setattr(model, name, old_monitor)
    del model.monitor

    if transfer_experience:
        monitor = Monitor.get_monitor(model)
        assert monitor is not old_monitor
        monitor._num_batches_seen = old_monitor._num_batches_seen
        monitor._examples_seen = old_monitor._examples_seen
        monitor._epochs_seen = old_monitor._epochs_seen

    return model

def read_channel(model, channel_name, monitor_name = 'monitor'):
    return getattr(model, monitor_name).channels[channel_name].val_record[-1]

def get_channel(model, dataset, channel, cost, batch_size):
    """
    Make a temporary monitor and return the value of a channel in it.

    model: A pylearn2.models.model.Model instance. Will evaluate the
           channel for this Model.
    dataset: The Dataset to run on
    channel: A string identifying the channel name to evaluate
    cost: The Cost to setup for monitoring
    batch_size: The size of the batch to use when running the monitor

    returns the value of the requested channel.

    Note: this doesn't modify the model (unless some of the channel prereqs do).
          In particular, it does not change model.monitor.
    """
    monitor = Monitor(model)
    monitor.setup(dataset=dataset, cost=cost, batch_size=batch_size)
    monitor()
    channels = monitor.channels
    channel = channels[channel]
    val_record = channel.val_record
    value ,= val_record
    return value

_err_no_data = "You tried to add a channel to a Monitor that has no dataset."
_err_ambig_data = ("You added a channel to a Monitor that has multiple datasets, "
        "and did not specify which dataset to use it with.")
