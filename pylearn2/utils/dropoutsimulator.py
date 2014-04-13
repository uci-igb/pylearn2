# Author Peter Sadowski
# This function can be used to propagate data through a mlp, with dropout.
import theano
import pylearn2
import numpy as np
import matplotlib.pyplot as plt
import pylearn2.train_extensions.dropout_analysis
import pylearn2.datasets.mnist

def sigmoid(x):
      return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    sig = np.exp(x)
    return sig / np.sum(sig, axis=1)

def NWGM(X):
    # Compute NWGM along axis 0.
    G  = np.exp2(np.sum(np.log2(X), axis=0) * 1.0/X.shape[0])  # Geometric mean.
    N  = np.exp2(np.sum(np.log2(1.0-X), axis=0) * 1.0/X.shape[0])
    NWGM = G / (G + N)             # Normalized geometric mean.()
    return NWGM

def logit(x):
    '''
    Inverse logistic function.
    Maxes out at 30, -30 because this is what theano does.
    See theano/tensor/nnet/sigm.py
    '''
    x = np.maximum(x, 1e-20)
    x = np.minimum(x, 1.0-1e-20)
    x = np.log(x / (1.0 - x))
    x = np.maximum(x, -30.0)
    x = np.minimum(x, 30.0)
    return x

# Define dropout analysis simulation.
def Hinton_include_probs(model):
    # Default dropout parameters for MNIST. Include prob of 0.8 for input, 0.5 for all hidden layers.
    input_include_probs = {}
    for lname in model.layer_names:
        if lname == 'h0':
            p = 0.8
        else:
            p = 0.5
        input_include_probs[lname] = p
    return input_include_probs
    
def simulate(model, dataset, input_include_probs=1, batch_size=None, niter=0):

    if batch_size == None:
        # Use all.
        batch_size = dataset.X.shape[0]

    # Dataset
    if dataset=='MNIST':
        dataset = pylearn2.datasets.mnist.MNIST(which_set='train', one_hot=True, start=0, stop=60000)
    databatch = dataset.iterator(mode='sequential', batch_size=batch_size).next()
    if input_include_probs==1:
        input_include_probs = {lname: 1 for lname in model.layer_names}
    elif input_include_probs=='Hinton':
        # Used for Hinton's dropout paper.
        #input_include_probs = {'h0':0.8, 'h1':0.5, 'y':0.5} # Must match model layer names.
        #input_scales = {'h0':1.25, 'h1':2.0, 'y':2.0}
        input_include_probs = {lname: 0.8 if lname=='h0' else 0.5 for lname in model.layer_names}    
    input_scales = {lname: 1.0/p for lname,p in input_include_probs.iteritems()}

    # Load dropout analysis object. Compiles function for computing activation from dropout simulations as well as the final, nondropout, activation.
    da = pylearn2.train_extensions.dropout_analysis.Divergence(model, dataset, input_include_probs, input_scales)
    
    # Compute final activations without dropout.
    falist = [np.zeros((1, layer.output_space.dim, batch_size)) for layer in model.layers]
    activations = da.single_fprop(inputs=databatch, dropout=False)
    for l in range(len(model.layers)):
        a = activations[l].transpose()
        assert(a.shape[1] == batch_size)
        falist[l][0, :, :] = a.reshape((1,a.shape[0], batch_size))
    if niter==0:
        # No need to compute dropout simulations.
        return falist
    
    
    # Run dropout simulations, compute neuron activations.
    nlayers = len(model.layers) # We include input as a layer. 
    # Create list of 3d matrices to store all activations.
    alist = [np.zeros((niter, layer.output_space.dim, batch_size)) for layer in model.layers]
    for isample in range(niter):
        activations = da.single_fprop(inputs=databatch, dropout=True) # Should instead give data. batchxnodes
        for l in range(len(alist)):
            a = activations[l].transpose() # nodes x batch
            assert(a.shape[1] == batch_size)
            alist[l][isample, :, :] = a.reshape((1, a.shape[0], batch_size))
    
    # Compute input sums.
    #xlist = [logit(a) for a in alist]
    return alist, falist

def summary_stats(alist):
    # Compute summary stats from dropout activations returned by simulate.
    Elist = []
    NWGMlist = []
    Vlist = []
    
    #raise Exception('scipy not working!')
    #from scipy import stats as stats

    for l, a in enumerate(alist):
        E  = np.mean(a, axis=0)      # Arithmetic mean over dropout samples.
        #G  = stats.gmean(a, axis=0)  # Geometric mean.
        #G  = np.prod(a, axis=0) ** 1.0/a.shape[0]  # Geometric mean.
        G  = np.exp2(np.sum(np.log2(a), axis=0) * 1.0/a.shape[0])  # Geometric mean.
        #N  = stats.gmean(1.0-a, axis=0)
        #N  = np.prod(1.0-a, axis=0) ** 1.0/a.shape[0]
        N  = np.exp2(np.sum(np.log2(1.0-a), axis=0) * 1.0/a.shape[0])
        NWGM = G / (G + N)             # Normalized geometric mean.
        V = np.var(a, axis=0)

        # Change 1 x Units x Inputs matrix to Units x Inputs
        Elist.append(E)
        NWGMlist.append(NWGM)
        Vlist.append(V)
    return Elist, NWGMlist, Vlist

