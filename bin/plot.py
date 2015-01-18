#!/usr/bin/env python

import sys

import numpy as np
from matplotlib import pyplot as plt



def main():
    infile = open(sys.argv[1])
    data = np.array(map(lambda x:map(float, x.strip('\n').split('\t')), infile.readlines()))
    X = data[:, 0:-1]
    (N, D) = X.shape
    Y = data[:, -1].reshape((N, 1))
    
    plt.plot(X[np.where(Y == 0)[0]][:, 0], X[np.where(Y == 0)[0]][:, 1], 'b.')
    plt.plot(X[np.where(Y == 1)[0]][:, 0], X[np.where(Y == 1)[0]][:, 1], 'r.') 
    plt.show()
    
    infile.close()


if __name__ == '__main__':
    main()