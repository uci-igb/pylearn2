#!/usr/bin/sh

echo "Sourcing centos6 profile with py2.7.6"
#source /auto/igb-libs/linux/centos/6.x/x86_64/profiles/gpu_py2.6
source /auto/igb-libs/linux/centos/6.x/x86_64/profiles/gpu_py2.7.6

# Add pylearn2 path
PYLEARN2=/extra/pjsadows0/ml/improver2013/pylearn2
export PYTHONPATH=$PYLEARN2:$PYTHONPATH
export PYLEARN2_DATA_PATH=/extra/pjsadows0/ml/data

export PATH=$PYLEARN2/pylearn2/scripts:$PATH
export PYLEARN2_VIEWER_COMMAND='eog --new-instance'
# GPU settings
#export THEANO_FLAGS="device=gpu"
export THEANO_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32" # Important for using gpu.



