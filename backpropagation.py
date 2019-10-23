# =============================================================================
# 2019.10.08
# neural network (back propagation) to predict hand write number
# @author:DanielChungYi
# =============================================================================


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

#load data
mat_contents = sio.loadmat('ex4data1.mat')
x = mat_contents['X']
y = mat_contents['y']
m = len(x)
n = x.shape[1]
num_labels = 10
x0 = np.full((5000,1),1)
X = np.append(x0,x,axis = 1)

#load weights
mat_contents = sio.loadmat('ex4weights.mat')
theta1 = mat_contents['Theta1']
theta2 = mat_contents['Theta2']
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   (note that we have mapped "0" to label 10)
# =============================================================================
#    
# =============================================================================
Theta1_grad = np.zeros((np.shape(theta1)[0],np.shape(theta1)[1]))
Theta2_grad = np.zeros((np.shape(theta2)[0],np.shape(theta2)[1]))
#f
a1 = X
z2 = np.dot(a1,theta1.T)
a2 = sigmoid(z2)
a2_0 = np.full((5000,1),1)
a2 = np.append(a2_0,a2,axis = 1) 
z3 = np.dot(a2,theta2.T)
h_theta = sigmoid(z3)
y_new = np.zeros((num_labels, m))
for i in range(m):
    y_new[y[i,0]-1,i] = 1
#bp
for i in range(m):
    a1 = X[i,:]
    a1 = X[0:1,:]
    z2 = np.dot(theta1,a1.T)
    a2 = sigmoid(z2)
    a2_0 = np.zeros((1,1))
    a2 = np.append(a2_0,a2,axis = 0) 
    z3 = np.dot(theta2,a2)
    a3 = sigmoid(z3)
    y1 = y_new[:,0:1]
    delta_3 = a3 - y1
    z2_0 = np.zeros((1,1))
    z2 = np.append(z2_0,z2,axis = 0) 
    delta_2 = np.dot(delta_3.T,theta2) * sigmoidGradient(z2).T

delta_2 = delta_2[0:1,1:26]
Theta2_grad = Theta2_grad + delta_3 * a2.T; # (10*1)*(1*26)
Theta1_grad = Theta1_grad + delta_2.T * a1 # (25*1)*(1*401)
Theta2_grad = (1/m) * Theta2_grad; # (10*26)
Theta1_grad = (1/m) * Theta1_grad; # (25*401)
# =============================================================================
# 
# =============================================================================
#define sigmoid function
def sigmoid(input_x):
    return 1/(1+np.exp(-input_x))

def sigmoidGradient(input_x):
    return sigmoid(input_x) * (1 - sigmoid(input_x));

def train(theta1, theta2, X, y):
    Theta1_grad = np.zeros((np.shape(theta1)[0],np.shape(theta1)[1]))
    Theta2_grad = np.zeros((np.shape(theta2)[0],np.shape(theta2)[1]))
    #f
    a1 = X
    z2 = np.dot(a1,theta1.T)
    a2 = sigmoid(z2)
    a2_0 = np.full((5000,1),1)
    a2 = np.append(a2_0,a2,axis = 1) 
    z3 = np.dot(a2,theta2.T)
    h_theta = sigmoid(z3)
    y_new = np.zeros((num_labels, m))
    for i in range(m):
        y_new[y[i,0]-1,i] = 1
    #bp
    a1 = X[0:1,:]
    z2 = np.dot(theta1,a1.T)
    a2 = sigmoid(z2)
    a2_0 = np.zeros((1,1))
    a2 = np.append(a2_0,a2,axis = 0) 
    z3 = np.dot(theta2,a2)
    a3 = sigmoid(z3)
    y1 = y_new[:,0:1]
    delta_3 = a3 - y1
    z2_0 = np.zeros((1,1))
    z2 = np.append(z2_0,z2,axis = 0) 
    delta_2 = np.dot(delta_3.T,theta2) * sigmoidGradient(z2).T
    for i in range(m):
        a1 = X[i,:]
    delta_2 = delta_2[0:1,1:26]
    Theta2_grad = Theta2_grad + delta_3 * a2.T; # (10*1)*(1*26)
    Theta1_grad = Theta1_grad + delta_2.T * a1 # (25*1)*(1*401)
    Theta2_grad = (1/m) * Theta2_grad; # (10*26)
    Theta1_grad = (1/m) * Theta1_grad; # (25*401)

def randInitializeWeights(row,column):
    weight = np.zeros((row,column))
    return weight

#def unrollweight()
    
train()
