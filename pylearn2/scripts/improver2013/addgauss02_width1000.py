# Author: Peter Sadowski

import sys
import os
import theano
import pylearn2
import pylearn2.datasets.improver2013
import pylearn2.training_algorithms.sgd
import pylearn2.termination_criteria
import pylearn2.costs.mlp.dropout
import pylearn2.costs.mlp.addgauss
#import pylearn2.space
import pylearn2.models.mlp as mlp
import pylearn2.train

def init_train():
    # Initialize train object.
    idpath = os.path.splitext(os.path.abspath(__file__))[0] # ID for output files.
    save_path = idpath + '.pkl'

    # Dataset
    #seed = 42
    nvis = 32
    dataset_train = pylearn2.datasets.improver2013.IMPROVER2013(which_set='A_human_phospho_from_rat_phospho_replicate')
    
    # Parameters
    momentum_saturate = 500
    
    # Model
    model = pylearn2.models.mlp.MLP(layers=[mlp.Sigmoid(
                                                layer_name='h0',
                                                dim=1000,
                                                istdev=.01),
                                            mlp.Sigmoid(
                                                layer_name='y',
                                                dim=32,
                                                istdev=.001)
                                           ],
                                    nvis=nvis
                                    )

    # Algorithm
    algorithm = pylearn2.training_algorithms.sgd.SGD(
                    batch_size=1000,   # If changed, change learning rate!
                    learning_rate=.1, # In dropout paper=10 for gradient averaged over batch. Depends on batchsize.
                    init_momentum=.5, 
                    monitoring_dataset = {'train':dataset_train,
                                          
                                          },
                    termination_criterion = pylearn2.termination_criteria.EpochCounter(
                                                max_epochs=6000
                                            ),
                    #termination_criterion=pylearn2.termination_criteria.Or(criteria=[
                    #                        pylearn2.termination_criteria.MonitorBased(
                    #                            channel_name="valid_objective",
                    #                            prop_decrease=0.00001,
                    #                            N=40),
                    #                        pylearn2.termination_criteria.EpochCounter(
                    #                            max_epochs=momentum_saturate)
                    #                        ]),
                    #cost=pylearn2.costs.cost.SumOfCosts(
                    #    costs=[pylearn2.costs.mlp.Default()
                    #           ]
                    #),
                    #cost = pylearn2.costs.mlp.dropout.Dropout(
                    #    input_include_probs={'h0':1., 'h1':1., 'h2':1., 'h3':1., 'y':0.5},
                    #    input_scales={ 'h0': 1., 'h1':1., 'h2':1., 'h3':1., 'y':2.}),
                    cost = pylearn2.costs.mlp.addgauss.AddGauss(input_noise_stdev={ 'h0':0.2, 'y':0.2}),
                    update_callbacks=pylearn2.training_algorithms.sgd.ExponentialDecay(
                                        decay_factor=1.000004, # Decreases by this factor every batch. (1/(1.000001^8000)^100 
                                        min_lr=.000001
                                        )
                )
    # Extensions 
    extensions=[ 
        #pylearn2.train_extensions.best_params.MonitorBasedSaveBest(channel_name='train_y_misclass',save_path=save_path)
        pylearn2.training_algorithms.sgd.MomentumAdjustor(
            start=0,
            saturate=momentum_saturate,
            final_momentum=.99  # Dropout=.5->.99 over 500 epochs.
            )
        ]
    # Train
    train = pylearn2.train.Train(dataset=dataset_train,
                                 model=model,
                                 algorithm=algorithm,
                                 extensions=extensions,
                                 save_path=save_path,
                                 save_freq=100)
    return train
    
def train(mytrain):
    # Execute training loop.
    debug = False
    logfile = os.path.splitext(mytrain.save_path)[0] + '.log'
    print 'Using=%s' % theano.config.device # Can use gpus. 
    print 'Writing to %s' % logfile
    print 'Writing to %s' % mytrain.save_path
    sys.stdout = open(logfile, 'w')        
    mytrain.main_loop()


if __name__=='__main__':
    # Initialize and train.
    mytrain = init_train()
    train(mytrain) 
