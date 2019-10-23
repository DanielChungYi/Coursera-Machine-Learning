# =============================================================================
# 2019.10.04
# one vs all using Logistic Regression
# @author:DanielChungYi
# =============================================================================
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

mat_contents = sio.loadmat('ex3data1.mat')
x = mat_contents['X']
y = mat_contents['y']
m = len(x)
n = x.shape[1]
num_labels = 10

#define sigmoid function
def sigmoid(input_x):
    return 1/(1+np.exp(-input_x))

#define on vs all classifier
def oneVsAll(x,y,num_labels):
    all_theta = np.full((1,n+1),0)
    x0 = np.full((5000,1),1)
    X = np.append(x0,x,axis = 1)
    #gradinet 
    for i in range(num_labels):
    #set initail theta
        initial_theta =np.full((n+1,1),0) 
        for j in range(60000):
            error = sigmoid(np.dot(X,initial_theta)) - np.where(y==i,1,y*0)
            grad = (1 / m) * np.dot(error.T,X).T
            initial_theta = initial_theta - 0.01 * grad
            
        all_theta = np.append(all_theta,initial_theta.T,axis=0)
    return all_theta

finial_weight = oneVsAll(x,y,10)
finial_weight = np.delete(finial_weight,0,0)
# =============================================================================
# x0 = np.full((5000,1),1)
# X = np.append(x0,x,axis = 1)
# predict = sigmoid(np.dot(X[1:2,:],finial_weight.T))
# predict = np.where(predict>0.5,1,0)
# result = (predict == y)
# =============================================================================
