# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:01:42 2020

@author: Dawnborn
"""

# SVM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

#划分训练集测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#特征量化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transfrom(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
clf = SVC(kern = 'linear', random_state = 0)
clf.fit(X_train, y_train)
