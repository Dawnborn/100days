# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:05:17 2020

@author: Dawnborn
"""

# d1 数据预处理

import numpy as np
import pandas as pd

# 导入数据集-----------------------
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# 处理丢失数据, 平均值替换------------------------------
from sklearn.impute import SimpleImputer
imputer = imputer = SimpleImputer(missing_values=np.nan, strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# 将标签量转变成可处理的数字量--------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder() 
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#独热编码，将用于标签的数字转化为各自垂直的分量
from sklearn.compose import ColumnTransformer 
ct = ColumnTransformer([('Country', OneHotEncoder(),[0])], remainder='passthrough')
X = ct.fit_transform(X)

# Label编码，结果为连续的数值变量
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# 拆分训练集和测试集合-------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

# 标准化处理---------------------------------
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)