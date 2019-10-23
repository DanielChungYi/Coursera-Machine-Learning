# =============================================================================
# 2019.10.04
# neural network to predict hand write number
# @author:DanielChungYi
# =============================================================================

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

#load data
mat_contents = sio.loadmat('ex3data1.mat')
x = mat_contents['X']
y = mat_contents['y']
m = len(x)
n = x.shape[1]
num_labels = 10
x0 = np.full((5000,1),1)
X = np.append(x0,x,axis = 1)

#load weights
mat_contents = sio.loadmat('ex3weights.mat')
theta1 = mat_contents['Theta1']
theta2 = mat_contents['Theta2']
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   (note that we have mapped "0" to label 10)

#define sigmoid function
def sigmoid(input_x):
    return 1/(1+np.exp(-input_x))

def predict(Theta1, Theta2, X):
    a1 = X
    z2 = np.dot(a1,Theta1.T)
    a2 = sigmoid(z2)
    a2_0 = np.full((5000,1),1)
    a2 = np.append(a2_0,a2,axis = 1) 
    z3 = np.dot(a2,Theta2.T)
    a3 = sigmoid(z3)
    return a3

result = predict(theta1, theta2, X)

