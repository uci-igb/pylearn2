__authors__ = "Peter Sadowski"

from pylearn2.datasets import dense_design_matrix
import os
import numpy as np
#import pickle as pkl

datapath = '~/improver2013/out/'

def converter(x):
    if x == 'NA':
        return np.nan
    else:
        return float(x)

class IMPROVER2013(dense_design_matrix.DenseDesignMatrix):
    
    #def __init__(self, which_set, binarize=False, randomize=True, holdout=False, start=0, stop=10000): 
    def __init__(self, which_set, binarize=False, randomize=True, holdout=False, start=0, stop=np.inf): 
       
        fin_y = None
        if which_set == 'rat_train':
            fin_X = open(datapath + 'SBV_STC_subchallenge1/GEx_rat_train.txt', 'r')
            fin_y = open(datapath + 'SBV_STC_subchallenge1/GEx_rat_train.txt.output', 'r')
        elif which_set == 'rat_test':
            fin_X = open(datapath + 'SBV_STC_subchallenge1/GEx_rat_test.txt', 'r')
            fin_y = None
        elif which_set == 'human_train':
            fin_X = open(datapath + 'SBV_STC_subchallenge1/GEx_human_train.txt', 'r')
            fin_y = open(datapath + 'SBV_STC_subchallenge1/GEx_human_train.txt.output', 'r') 
            X = np.loadtxt(fin_X, skiprows=1, dtype='float32')
            y = np.loadtxt(fin_y, skiprows=1, dtype='float32')
        elif which_set == 'A_phospho':
            # Input is rat A phospho, output is human A phospho.
            fin_X = open(datapath + 'SBV_STC_subchallenge2/training.data/GEx_rat_train.txt.output', 'r')
            fin_y = open(datapath + 'SBV_STC_subchallenge2/training.data/GEx_human_train.txt.output', 'r') 
            X = np.loadtxt(fin_X, skiprows=1, usecols=range(1,33), dtype='float32')
            y = np.loadtxt(fin_y, skiprows=1, usecols=range(1,33), dtype='float32')
            #y = np.zeros((X.shape[0], 1))
        elif which_set == 'B_phospho':
            # Input is rat B phospho, output is human B phospho.
            fin_X = open(datapath + 'SBV_STC_subchallenge2/GEx_rat_test.txt.output', 'r')
            X = np.loadtxt(fin_X, skiprows=1, dtype='float32') # This older file does not contain row header.
            #X = np.loadtxt(fin_X, skiprows=1, usecols=range(1,33), dtype='float32')
            y = np.zeros((X.shape[0], 1))
            #fin_y = open(datapath + 'SBV_STC_subchallenge2/GEx_human_train.txt.output', 'r') 
        elif which_set == 'A_geneset':
            # Input is rat A_geneset, output is human A_geneset.
            fin_X = open(datapath + 'SBV_STC_subchallenge3/training.data/GEx_rat_train.txt.fdr', 'r')
            fin_y = open(datapath + 'SBV_STC_subchallenge3/training.data/GEx_human_train.txt.fdr', 'r')
            converters = dict((i, converter) for i in range(1,247))
            X = np.loadtxt(fin_X, skiprows=1, usecols=range(1,247), converters=converters, dtype='float32')
            y = np.loadtxt(fin_y, skiprows=1, usecols=range(1,247), converters=converters, dtype='float32')
            X = X[~np.isnan(X).any(axis=1)]
            y = y[~np.isnan(y).any(axis=1)]
        elif which_set == 'B_geneset':
            # Input is rat B_geneset, output is human B_geneset.
            fin_X = open(datapath + 'SBV_STC_subchallenge3/GEx_rat_test.txt.fdr', 'r')
            X = np.loadtxt(fin_X, skiprows=1, dtype='float32')
            #X = np.loadtxt(fin_X, skiprows=1, usecols=range(1,246), dtype='float32')
            y = np.zeros((X.shape[0], 1))
        elif which_set == 'A':
            fin_X_1 = open(datapath + 'SBV_STC_subchallenge1/GEx_rat_train.txt', 'r')
            fin_X_2 = open(datapath + 'SBV_STC_subchallenge1/GEx_rat_train.txt.output', 'r')
            fin_y_1 = open(datapath + 'SBV_STC_subchallenge1/GEx_human_train.txt', 'r')
            fin_y_2 = open(datapath + 'SBV_STC_subchallenge1/GEx_human_train.txt.output', 'r') 
            # Need to concatenate these features.
            raise Error

        # Format data.
        #X = np.loadtxt(fin_X, skiprows=1, dtype='float32')
        #if fin_y == None:
            # No labels for test data. 
        #    y = np.zeros((X.shape[0], 1))
        #else:
        #    y = np.loadtxt(fin_y, skiprows=1, dtype='float32')
        #X = X.astype(np.float32)
        #y = y.astype(np.int)
        #y = np.reshape(y, (-1,1)) # 
        #X = X / 255. # Data file is 0-255

        # Binary
        if binarize:
            # Genes are either on or off.
            X = X.clip(3.0, 7.0)
            X = (X-3.0)/4.0
            y = y.clip(0.0, 4.0)
            y = y / 4.0

        # Holdout data.
        if holdout:
            assert(which_set == 'rat_train')
            fin = open(datapath + 'SBV_STC_subchallenge1/GEx_rat_train.txt.sample_groups', 'r')
            c = np.loadtxt(fin, skiprows=0, dtype='string')
            #cv1 = ['PROKINECITIN2', 'IFNG', 'HIGHGLU', 'TNFA', 'PDGFB']
            cv2 = ['ODN2006', 'FLAST', 'AMPHIREGULIN', 'X5AZA', 'CHOLESTEROL'] 
            validlist = cv2
            valididx = [i for i,cond in enumerate(c) if cond in validlist]
            trainidx = [i for i,cond in enumerate(c) if cond not in validlist]
            assert(len(valididx) + len(trainidx) == len(c))
            if holdout=='train':
                X = X[trainidx, :]
                y = y[trainidx, :]
            elif holdout=='valid':
                X = X[valididx, :]
                y = y[valididx, :]
            else:
                raise Exception
            

        # Randomize data.
        assert X.shape[0] == y.shape[0]
        stop = min(stop, X.shape[0])
        if randomize:
            rng = np.random.RandomState(42)  # reproducible results with a fixed seed
            indices = np.arange(X.shape[0])
            rng.shuffle(indices)
            X = X[indices, :]
            y = y[indices, :]

        # Limit number of samples.
        X = X[start:stop, :]
        y = y[start:stop, :]
        
        super(IMPROVER2013, self).__init__(X=X, y=y)
        

