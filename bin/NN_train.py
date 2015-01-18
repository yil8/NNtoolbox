#!/usr/bin/env python

import sys
import pickle as pkl

import numpy as np

from utils import *

def main():
    infile = open(sys.argv[1])
    outfile = open(sys.argv[2], 'w')
    epoch_num = int(sys.argv[3])
    lr_W = float(sys.argv[4])
    lr_b = float(sys.argv[5])

    data = np.array(map(lambda x:map(float, x.strip('\n').split('\t')), infile.readlines()))
    X = data[:, 0:-1]
    (N, D) = X.shape
    Y = data[:, -1].reshape((N, 1))

    W0 = np.random.random((D, D))
    W1 = np.random.random((D, D))
    w = np.random.random((D, 1))
    
    B0 = np.random.random((D, 1))
    B1 = np.random.random((D, 1))
    b = np.random.random((1))
    
    data_args = (X, Y)
    hyper_args = (lr_W, lr_b)
    
    para_args = (W0, W1, w, B0, B1, b)
    
    
    for k in range(0, epoch_num):
        para_args = epoch_(data_args, para_args, hyper_args)
        Y_hat, error_sum = predict_(data_args, para_args)
        
        print "epoch %s" %(k)
        print "error/sum : %s/%s" %(error_sum, N)
        print 'W0: %s' % para_args[0]
        print 'W1: %s' % para_args[1]
        print 'w: %s' % para_args[2]
        print 'B0: %s' % para_args[3]
        print 'B1: %s' % para_args[4]
        print 'b: %s' % para_args[5]

    pkl.dump(para_args, outfile)
        
    infile.close()
    outfile.close()
        
    
if __name__ == '__main__':
    main()