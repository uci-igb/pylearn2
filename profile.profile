#!/usr/bin/sh

# Add pylearn2 path
PYLEARN2=~/improver2013/improver2013/opt/pylearn2
export PYTHONPATH=$PYLEARN2:$PYTHONPATH
export PYLEARN2_VIEWER_COMMAND='eog --new-instance'
# GPU settings
export THEANO_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32" # Important for using gpu.



