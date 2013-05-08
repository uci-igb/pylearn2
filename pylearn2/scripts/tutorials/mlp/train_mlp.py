# This is a non-notebook version of the multilayer_perceptron tutorial. -Peter Sadowski

import theano
print 'Using: %s' % theano.config.device # Can use gpus. 

# The first example uses 2 layers, sigmoidal units, and minibatch with nonlinear CG
from pylearn2.config import yaml_parse
train_3 = yaml_parse.load(open('dropout.yaml', 'r'))
#train_3 = yaml_parse.load(open('mlp.yaml', 'r'))
train_3.main_loop()
#import os
#os.system('show_weights.py mlp_best.pkl')

# The second examples uses 3 layers, a rectified linear layer, and minibatch SGD
#from pylearn2.config import yaml_parse
#train_2 = yaml_parse.load(open('mlp2.yaml', 'r')
#train_2.main_loop()
# print_monitor script, the following works in ipython?
#!print_monitor.py mlp_3_best.pkl | grep test_y_misclass
