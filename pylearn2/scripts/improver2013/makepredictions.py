# Script for producing predictions from a trained nn.
# Author Peter Sadowski 2013

import os
os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cpu,floatX=float32" # gpu0,etc.
import theano
print 'Using: %s' % theano.config.device
import pylearn2
import pickle as pkl
from pylearn2.config import yaml_parse
import numpy as np
import pylearn2.datasets.improver2013 as improver2013
#import pylearn2.datasets.autoencode as autoencode
import pylab
import pylearn2.utils.dropoutsimulator as simulator


model = pkl.load(open('addgauss02_width1000.pkl', 'r'))

# Predict on test set.
traindata = improver2013.IMPROVER2013(which_set='A_phospho', binarize=True, randomize=False, holdout=False)
data = improver2013.IMPROVER2013(which_set='B_phospho', binarize=True, randomize=False, holdout=False)

trainpred = simulator.simulate(model, dataset=traindata)
trainpred = trainpred[-1][0,:,:].T
#np.savetxt('/home/baldig/projects/genomics/improver2013/out/SBV_STC_subchallenge1/transformed/GEx_rat_train.txt.predicted', trainpred, fmt='%0.12f')

falist = simulator.simulate(model, dataset=data)
testpred = falist[-1][0,:,:].T

# Input
np.savetxt('GEx_rat_train.txt.output', traindata.X, fmt='%0.12f') 
np.savetxt('GEx_rat_test.txt.output', data.X, fmt='%0.12f') 
# Output
np.savetxt('GEx_human_train.txt.output', traindata.y, fmt='%0.12f') 
np.savetxt('GEx_human_train.txt.predicted', trainpred, fmt='%0.12f')
np.savetxt('GEx_human_test.txt.predicted', testpred, fmt='%0.12f') 


