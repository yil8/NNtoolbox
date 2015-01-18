import numpy as np



def sigmoid(X):
    
    return 1/(1 + np.exp(-1*X))

def gradient_w(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    I1, I2, O1, O2, s, y_hat = infer_args
    
    g_w = 2*(y_hat - y)*sigmoid(s)*(1 - sigmoid(s))*O2
    
    return g_w

def gradient_W1(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    w, W0, W1 = para_args
    I1, I2, O1, O2, s, y_hat = infer_args
    
    temp = 2*(y_hat - y)*sigmoid(s)*(1 - sigmoid(s))*w*sigmoid(I2)*sigmoid(1 - I2)
    g_W1 = np.dot(temp, O1.transpose())
    
    return g_W1

def gradient_W0(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    w, W0, W1 = para_args
    I1, I2, O1, O2, s, y_hat = infer_args
    
    temp = 2*(y_hat - y)*sigmoid(s)*(1 - sigmoid(s))*w*sigmoid(I2)*sigmoid(1 - I2)
    temp = np.dot(W1.transpose(), temp)
    temp = temp*sigmoid(I1)*(1 - sigmoid(I1))
    g_W = np.dot(temp, O0.transpose())
    
    return g_W1

def predict_x(x, para_args):
    w, W0, W1 = para_args

    O0 = x
    I1 = np.dot(W0, O0)
    O1 = sigmoid(I1)
    I2 = np.dot(W1, O1)
    O2 = sigmoid(I2)
    s = np.dot(w.transpose(), O2)
    y_hat = sigmoid(s)
    
    return y_hat

def forward(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    w, W0, W1 = para_args
    
    I1 = np.dot(W0, O0)
    O1 = sigmoid(I1)
    I2 = np.dot(W1, O1)
    O2 = sigmoid(I2)
    s = np.dot(w.transpose(), O2)
    y_hat = sigmoid(s)
    
    infer_args = (I1, I2, O1, O2, s, y_hat)

    return infer_args

def backward(data_args, para_args, infer_args, hyper_args):
    w, W0, W1 = para_args
    l_rate = hyper_args
    
    w = w - l_rate*gradient_w(data_args, para_args, infer_args, hyper_args)
    W1 = W1 - l_rate*gradient_W1(data_args, para_args, infer_args, hyper_args)
    W0 = W0 - l_rate*gradient_W1(data_args, para_args, infer_args, hyper_args)

    para_args = (w, W0, W1)
    
    return para_args

def predict(data_args, para_args):
    X, Y = data_args

    (N, D) = X.shape
    Y_hat = np.random.random((N, 1))
    
    for i in range(0, N):
        x = X[i, :].reshape((D, 1))
        y_hat = predict_x(x, para_args)
        Y_hat[i] = y_hat
    
    error_sum = int(np.absolute(Y_hat - Y).sum())
    
    return (Y_hat, error_sum)

def epoch(data_args, para_args, hyper_args):
    X, Y = data_args
    (N, D) = X.shape
    infer_args = None
    
    for i in range(0, N):
        O0 = X[i, :].reshape((D, 1))
        y = Y[i]
        data_args_i = (O0, y)
        
        infer_args = forward(data_args_i, para_args, infer_args, hyper_args)
        para_args = backward(data_args_i, para_args, infer_args, hyper_args)
        
    return para_args


#########################################################################################
def gradient_W0_(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    W0, W1, w, B0, B1, b = para_args
    I1, I2, O1, O2, s, y_hat = infer_args
     
    temp = 2*(y_hat - y)*sigmoid(s)*(1 - sigmoid(s))*w*sigmoid(I2)*sigmoid(1 - I2)
    temp = np.dot(W1.transpose(), temp)
    temp = temp*sigmoid(I1)*(1 - sigmoid(I1))
    g_W0 = np.dot(temp, O0.transpose())
    
    
    return g_W0

def gradient_B0_(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    W0, W1, w, B0, B1, b = para_args
    I1, I2, O1, O2, s, y_hat = infer_args
    
    temp = 2*(y_hat - y)*sigmoid(s)*(1 - sigmoid(s))*w*sigmoid(I2)*sigmoid(1 - I2)
    temp = np.dot(W1.transpose(), temp)
    g_B0 = temp*sigmoid(I1)*sigmoid(1 - I1)
    
    return g_B0


def gradient_W1_(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    W0, W1, w, B0, B1, b = para_args
    I1, I2, O1, O2, s, y_hat = infer_args
    
    temp = 2*(y_hat - y)*sigmoid(s)*(1 - sigmoid(s))*w*sigmoid(I2)*sigmoid(1 - I2)
    g_W1 = np.dot(temp, O1.transpose())
    
    return g_W1

def gradient_B1_(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    W0, W1, w, B0, B1, b = para_args
    I1, I2, O1, O2, s, y_hat = infer_args
    
    g_B1 = 2*(y_hat - y)*sigmoid(s)*(1 - sigmoid(s))*w*sigmoid(I2)*sigmoid(1 - I2)
    
    return g_B1

def gradient_w_(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    I1, I2, O1, O2, s, y_hat = infer_args
    
    g_w = 2*(y_hat - y)*sigmoid(s)*(1 - sigmoid(s))*O2
    
    return g_w

def gradient_b_(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    I1, I2, O1, O2, s, y_hat = infer_args
    
    g_b = 2*(y_hat - y)*sigmoid(s)*(1 - sigmoid(s))
    
    return g_b
    

def forward_(data_args, para_args, infer_args, hyper_args):
    O0, y = data_args
    W0, W1, w, B0, B1, b = para_args

    I1 = np.dot(W0, O0) + B0
    O1 = sigmoid(I1)
    I2 = np.dot(W1, O1) + B1
    O2 = sigmoid(I2)
    s = np.dot(w.transpose(), O2) + b
    y_hat = sigmoid(s)
    
    infer_args = (I1, I2, O1, O2, s, y_hat)

    return infer_args

def backward_(data_args, para_args, infer_args, hyper_args):
    W0, W1, w, B0, B1, b = para_args
    lr_W, lr_b = hyper_args

    W0 = W0 - lr_W*gradient_W0_(data_args, para_args, infer_args, hyper_args)
    W1 = W1 - lr_W*gradient_W1_(data_args, para_args, infer_args, hyper_args)
    w = w - lr_W*gradient_w_(data_args, para_args, infer_args, hyper_args)
    B0 = B0 - lr_b*gradient_B0_(data_args, para_args, infer_args, hyper_args)
    B1 = B1 - lr_b*gradient_B1_(data_args, para_args, infer_args, hyper_args)
    b = b - lr_b*gradient_b_(data_args, para_args, infer_args, hyper_args)

    para_args = (W0, W1, w, B0, B1, b)
    
    return para_args

def predict_x_(x, para_args):
    W0, W1, w, B0, B1, b = para_args

    O0 = x
    I1 = np.dot(W0, O0) + B0
    O1 = sigmoid(I1)
    I2 = np.dot(W1, O1) + B1
    O2 = sigmoid(I2)
    s = np.dot(w.transpose(), O2) + b
    y_hat = sigmoid(s)
    
    return y_hat

def predict_(data_args, para_args):
    X, Y = data_args

    (N, D) = X.shape
    Y_hat = np.random.random((N, 1))
    
    for i in range(0, N):
        x = X[i, :].reshape((D, 1))
        y_hat = predict_x_(x, para_args)
        Y_hat[i] = y_hat
    
    error_sum = int(np.absolute(Y_hat - Y).sum())
    
    return (Y_hat, error_sum)

def epoch_(data_args, para_args, hyper_args):
    X, Y = data_args
    (N, D) = X.shape
    infer_args = None
    
    for i in range(0, N):
        O0 = X[i, :].reshape((D, 1))
        y = Y[i]
        data_args_i = (O0, y)
        
        infer_args = forward_(data_args_i, para_args, infer_args, hyper_args)
        para_args = backward_(data_args_i, para_args, infer_args, hyper_args)
        
    return para_args



#######################################################################
def predict_x_L(x, para_args, L):
    W0, W1, w, B0, B1, b = para_args
    
    if L == 3:
        pass
    
    O0 = np.array(x)
    I1 = np.dot(W0, O0) + B0
    O1 = sigmoid(I1)
    
    if L == 2:
        O1 = np.array(x)
    
    I2 = np.dot(W1, O1) + B1
    O2 = sigmoid(I2)
    
    if L == 1:
        O2 = np.array(x)
    
    s = np.dot(w.transpose(), O2) + b
    y_hat = sigmoid(s)
    
    return y_hat



def predict_L(data_args, para_args, L):
    X, Y = data_args

    (N, D) = X.shape
    Y_hat = np.random.random((N, 1))
    
    for i in range(0, N):
        x = X[i, :].reshape((D, 1))
        y_hat = predict_x_L(x, para_args, L)
        Y_hat[i] = y_hat
    
    error_sum = int(np.absolute(Y_hat - Y).sum())
    
    return (Y_hat, error_sum)
