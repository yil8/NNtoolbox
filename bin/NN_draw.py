#!/usr/bin/env python

import sys

import numpy as np
from matplotlib import pyplot as plt

RESOLUTION = 100
# W_RANGE = 1
# B_RANGE = 1
D = 2
MAP_RADIUS = 1

def sigmoid(X):
    
    return 1/(1 + np.exp(-1*X))


def get_grid(map_radius):
    R = (np.array(range(0, map_radius*RESOLUTION*2 + 1)) - map_radius*RESOLUTION)*1.0/RESOLUTION
    R_reversed = R[::-1]
    
    grid = []
    for j in range(0, len(R)):
        for i in range(0, len(R)):
            grid.append([R[i], R_reversed[j]])
    
    grid = np.array(grid)
    
    return grid

def NN_forward_i(x, parameters, depth, width):
    if depth == 0:
        w, b = parameters
        s = np.dot(w.transpose(), x) + b
        y = sigmoid(s)
        return y
    else:
        W_I, W_H, w, B_I, B_H, b = parameters
        I = np.dot(W_I, x) + B_I
        O = sigmoid(I)
        
        for l in range(0, depth-1):
            I = np.dot(W_H[l], O) + B_H[l]
            O = sigmoid(I)
        
        s = np.dot(w.transpose(), O) + b
        y = sigmoid(s)
        return y

def NN_forward(grid, parameters, depth, width):
    N, _ = grid.shape
    
    intensity = []
    for i in range(0, N):
        x_i = grid[i, :].reshape((D, 1))
        y_i = NN_forward_i(x_i, parameters, depth, width)
        intensity.append(y_i)
    
    intensity = np.array(intensity)
    
    return intensity

def get_parameters(depth, width, W_range, B_range):
    if depth == 0:
        w = np.random.random((D, 1))*W_range*2 - W_range
        b = np.random.random((1))*B_range*2 - B_range
        parameters = (w, b)
        
        return parameters
    else:
        W_I = np.random.random((width, D))*W_range*2 - W_range
        B_I = np.random.random((width, 1))*B_range*2 - B_range
        w = np.random.random((width, 1))*W_range*2 - W_range
        b = np.random.random((1))*B_range*2 - B_range
        
        W_H = []
        B_H = []
        
        for l in range(0, depth-1):
            W = np.random.random((width, width))*W_range*2 - W_range
            B = np.random.random((width, 1))*B_range*2 - B_range
            
            W_H.append(W)
            B_H.append(B)
        
        parameters = (W_I, W_H, w, B_I, B_H, b)
        
        return parameters

def main():
    hidden_layer_depth = int(sys.argv[1])
    hidden_layer_width = int(sys.argv[2])
    W_range = float(sys.argv[3])
    B_range = float(sys.argv[4])
    
    ticker_num = MAP_RADIUS*RESOLUTION*2 + 1
    grid = get_grid(MAP_RADIUS)

    parameters = get_parameters(hidden_layer_depth, hidden_layer_width, W_range, B_range)
    
    
    intensity = NN_forward(grid, parameters, hidden_layer_depth, hidden_layer_width)
    intensity = intensity.reshape((ticker_num, ticker_num))
    
    plt.figure(figsize=(15,15))
    plt.xticks([0, MAP_RADIUS*RESOLUTION*2], [-1*MAP_RADIUS, MAP_RADIUS])
    plt.yticks([0, MAP_RADIUS*RESOLUTION*2], [MAP_RADIUS, -1*MAP_RADIUS])
    plt.imshow(intensity.reshape((ticker_num, ticker_num)))
    plt.colorbar()
    plt.show()

    
    
    
    
    
if __name__ == '__main__':
    main()