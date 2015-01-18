#!/usr/bin/env python

import sys
import pickle as pkl

import numpy as np

from utils import *

def predict(data_args, para_args):
    X, Y = data_args

    (N, D) = X.shape
    Y_hat = np.random.random((N, 1))
    
    for i in range(0, N):
        x = X[i, :].reshape((D, 1))
        y_hat = predict_x_(x, para_args)
        Y_hat[i] = y_hat
    
    error_sum = int(np.absolute(Y_hat - Y).sum())
    
    return (Y_hat, error_sum)


def main():
    infile = open(sys.argv[1])
    inmodel = open(sys.argv[2])
    outfile = open(sys.argv[3], 'w')
    
    data = np.array(map(lambda x:map(float, x.strip('\n').split('\t')), infile.readlines()))
    X = data[:, 0:-1]
    (N, D) = X.shape
    Y = data[:, -1].reshape((N, 1))
    
    data_args = (X, Y)
    para_args = pkl.load(inmodel)
    Y_hat, error_sum = predict_(data_args, para_args)
    Y_predict = np.round(Y_hat)
    data[:, -1] = Y_predict.flatten()
    
    print "error/sum : %s/%s" %(error_sum, N)

    for i in range(0, N):        
        outfile.write('\t'.join(map(str, data[i, :])) + '\n')
        
        
    infile.close()
    inmodel.close()
    outfile.close()
        
    
if __name__ == '__main__':
    main()