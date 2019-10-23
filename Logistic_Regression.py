# =============================================================================
# 2019.10.02
# Logistic Regression
# @author:DanielChungYi
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model  import LogisticRegression
import matplotlib.pyplot as plt
df = pd.read_csv('ex2data1.txt')


# =============================================================================
# data frame 的一些用法
# print(df)
# print(df.shape)
# print("---")  
# print(df.columns) # 回傳欄位名稱  
# print("---")  
# print(df.index) # 回傳 index  
# print("---")  
# print(df.info) # 回傳資料內容  
# 
# print(df.iloc[:,0]) # 第一欄 
# print(df.iloc[:,1]) # 第二欄 
# print(df.iloc[:,2]) # 第三欄 
# 
# # 轉 np array:
# df[0].as_matrix()
#    
# # 轉list：
# df['0'].values.tolist()
# =============================================================================

# put colum 1 into data
z =df.columns
col = np.array([z[0] ,z[1], z[2]])
for i in col:
    i = float(i)
col=col[:,np.newaxis]
col = col.T
df_ = df.as_matrix()
df_ = np.append(df_,col,axis = 0)
#當axis為0時，陣列是加在下面（列數要相同）
#當axis為1時，陣列是加在右邊（行數要相同）

# =============================================================================
# #data from data frame
# x = df.iloc[:,0:2]
# x = x.as_matrix()
# x = np.array(x)
# y = df.iloc[:,2]
# y = y.as_matrix()
# =============================================================================

#data from numpy array
x = df_[:,0:2]
y = df_[:,2]
x0 = np.ones(100)
x0=x0[:,np.newaxis]
y=y[:,np.newaxis]
x = x.astype(float)
y = y.astype(float)
x = np.append(x0,x,axis = 1)
scores = np.zeros(100)
scores=scores[:,np.newaxis]
weights = np.random.randn(3,1)
make_plot=[]
m = len(x)

#define sigmoid function
def sigmoid(input_x):
    return 1/(1+np.exp(-input_x))

#define cost function
def cost(features,target,weights):
    scores=np.dot(features,weights)
    l1=np.sum(target*scores-np.log(1+np.exp(scores)))
    return l1

#gradient descent
for step in range(600000):
    scores=np.dot(x,weights)
    predictions=sigmoid(scores)
    output_error_singal=predictions - y
#    grad = (1 / m) * (h_theta - y)' * X;
    gradient=(1 / m) * np.dot((predictions - y).T,x).T
    weights = weights - 0.001*gradient
    make_plot.append(cost(x,y,weights))
    
#plot cost function
plt.plot(make_plot)

#plot
pos = np.where(y == 1)
neg = np.where(y == 0)
pos = pos[0]
neg = neg[0]
#pos_x = np.full((len(pos),1),0)
#pos_y = np.full((len(pos),1),0)
t = np.linspace(20,100,num=100)
for i in pos:
   plt.scatter(x[i,1],x[i,2],c='green')
for i in neg:
   plt.scatter(x[i,1],x[i,2],c='red')
#plt.plot(x[:,1],weights[0,0]*x[:,0]+weights[1,0]*x[:,1]+weights[2,0]*x[:,2],'b',linewidth=5)
plt.plot(t,-weights[0,0]/weights[1,0]-(weights[2,0]/weights[1,0])*t,'b')
plt.show()  

#predict
predict = sigmoid(np.dot(x,weights))
predict = np.where(predict>0.5,1,0)
result = (predict == y)
predict_rate = (np.sum(result==1) / 100)

#用sklearn驗證
clf=LogisticRegression(fit_intercept=True,C=1e15)
clf.fit(x,y)
print(clf.intercept_,clf.coef_)
print(clf.score(x,y))

