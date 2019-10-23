# =============================================================================
# 2019.10.01
# Linear Regression
# author:DanielChung
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(100)
y = 10.5 * x + 3.7 + np.random.randn(100)
f1 = plt.figure(1) 
plt.scatter(x,y,s=100)
plt.xlabel('x axis',fontsize=20)
plt.ylabel('y axis',fontsize=20,rotation=0)
plt.savefig('data_point')

m = len(y)
#print(m)
lr = 0.001
theta = np.random.randn(2,1)
iter_max = 100000

#add x0
x0 = np.ones(100)
x = np.vstack((x,x0))
x = x.T
y = y.T
y=y[:,np.newaxis]

#cost function
cost = (1/(2*m))*(np.dot(x,theta)-y)
cost = np.square(cost)
print(cost.shape)

#gradient descent
z =np.dot((np.dot(x,theta) - y).T,x)

for i in range(iter_max):
    theta = theta -lr * (1/m) * np.dot((np.dot(x,theta) - y).T,x).T
 #  theta = theta - alpha * (1/m) * (((X*theta) - y)' * X)'; % Vectorized  matlab
print(theta)


#plot
#plt.plot(x,y,'o',markersize=15,alpha=0.3)
plt.plot(x,theta[1]+theta[0]*x,'r',linewidth=5)
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20,rotation=0)