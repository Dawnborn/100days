# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:47:39 2020

@author: Dawnborn
"""

# 多元线性化

import pandas as pd
import numpy as np

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#将类别数据数字化
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
ct = ColumnTransformer( [('State', OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]    # 注意虚拟变量陷阱，实际上两位即可表达出3个不同城市，此处删去第0列

# 拆分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

# 绘图
import matplotlib.pyplot as plt
plt.scatter