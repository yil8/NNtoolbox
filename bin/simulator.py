#!/usr/bin/env python

import sys

import numpy as np

def background(R):
    x_1, x_2 = (np.random.rand(2) * 2*R) - R
    
    return (x_1, x_2)



def circle(c_x_1, c_x_2, r, label):
    while True:
        x_1, x_2 = (np.random.rand(2) - 0.5)*2*r
        if np.sqrt(x_1*x_1 + x_2*x_2) < r and label == 0:
            break
        elif np.sqrt(x_1*x_1 + x_2*x_2) > r and label == 1:
            break
            
    x_1 += c_x_1
    x_2 += c_x_2

    return (x_1, x_2)

def line(a, b, label):
    while True:
        x_1, x_2 = (np.random.rand(2) * 1000) - 500
        if x_1*a + b < x_2 and label == 0:
            break
        elif x_1*a + b > x_2 and label == 1:
            break

    return (x_1, x_2)


def bound(a, b, c, d, label):
    while True:
        x_1, x_2 = (np.random.rand(2) * 10) - 5
        if x_2 < a and label == 0:
            break
        elif x_2 > a and x_2 < b and label == 1:
            break
        elif x_2 > b and x_2 < c and label == 0:
            break
        elif x_2 > c and x_2 < d and label == 1:
            break
        elif x_2 > d and label == 1:
            break

    return (x_1, x_2)
    

def main():
#     outfile = open(sys.argv[1], 'w')
    N = int(sys.argv[1])
    label = int(sys.argv[2])
    
    for i in range(0, N):
        #x_1, x_2 = circle(0, 0, x, label)
        #x_1, x_2 = line(1, 200, label)
        #x_1, x_2 = background(10)
        x_1, x_2 = bound(-3, -1, 5, 5, label)
        
        print '\t'.join([str(x_1), str(x_2), str(label)])
        
        #outfile.write('\t'.join([str(x_1), str(x_2), str(label)]) + '\n')
    
    
    
    
    
#     outfile.close()


if __name__ == '__main__':
    main()