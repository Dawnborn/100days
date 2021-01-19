# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:25:13 2020

@author: Dawnborn
"""

# logistic regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:,4]

# 划分训练集和测试集-------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# 特征缩放------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 逻辑回归模型----------------------------
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

# 预测---------------------------------------
Y_pred = classifier.predict(X_test)

#评估-----------------------------------------
from sklearn.metrics import confusion_matrix # 混淆矩阵
cm =confusion_matrix(Y_test, Y_pred)

# 可视化-------------------------------------
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
# 绘制背景图像
x1,x2 = np.meshgrid( np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),
                 np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01) )
# np.arrange 生成指定始终点和步长的序列

plt.contourf( x1, x2, classifier.predict( np.array( [x1.ravel(), x2.ravel()] ).T ).reshape(x1.shape),
            alpha = 0.25, cmap = ListedColormap(('red', 'green')) )     #坐标矩阵转换成一维向量后计算结果，最后再转换为原形式
# ravel()将高维变一维

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
# 设置图像边界

for i, j in enumerate(np.unique(Y_set), start=0):    # enumerate 函数为可遍历对象生成索引, unique 派出重复项, 循环范围实质上就是（0,0） （1,1）
# for i, j in [(0,0), (1,1)]:
    plt.scatter(X_set[Y_set==j, 0], X_set[Y_set==j, 1], c = ListedColormap(('red', 'green'))(i), label=j)

plt. title(' LOGISTIC(Test set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt.legend()
plt.show()
    