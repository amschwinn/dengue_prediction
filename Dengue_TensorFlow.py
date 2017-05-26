# -*- coding: utf-8 -*-
"""
By: Austin Schwinn
Date: April 26, 2017
Subject: Predicting mosquito cause dengue
outbreaks in Puerto Rico and Peru.

Seperate tensorflow modeling from datawrangling done in R
"""
#%%
import pandas as pd
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split

#set our working directory
#os.chdir(os.path.dirname('__file__'))

#%%
#Load in existing DFs exported as CSV from R
iq = pd.read_csv("iq_features.csv")
sj = pd.read_csv("sj_features.csv")
w2_iq = pd.read_csv("w2_iq_features.csv")
w2_sj = pd.read_csv("w2_sj_features.csv")

#list of DF's
cities = [sj,iq,w2_sj,w2_iq]

#Remove R cols
for i in cities:
    del i[i.columns[0]]

#%%
#Create tensors for each feature column
features = sj.drop(sj.columns[(len(sj.columns)-2):(len(sj.columns))],
                   axis=1).drop(sj.columns[0:4],axis=1)
features = [i for i in features]
tensors = {}
for i in features:
    tensors[i] = tf.contrib.layers.real_valued_column(i)

#Now the same for the 2 week look back
w2_features = w2_sj.drop(w2_sj.columns[(len(w2_sj.columns)-1)],
                         axis=1).drop(w2_sj.columns[0:4],axis=1)
w2_features = [i for i in w2_features]
w2_tensors = {}
for i in w2_features:
    w2_tensors[i] = tf.contrib.layers.real_valued_column(i)

#Prediction value
label = ["total_cases"]    

#%%
#Create input functions
def input_fun(data_set):
    feature_cols = {k: tf.constant(data_set[k].values)
        for k in features}
    labels = tf.constant(data_set[label].values)
    return feature_cols, labels

#same for w2
def w2_input_fun(data_set):
    w2_feature_cols = {k: tf.constant(data_set[k].values)
        for k in w2_features}
    w2_labels = tf.constant(data_set[label].values)
    return w2_feature_cols, w2_labels

#%%
#split into training and test
np.random.seed(1991)
#Split original by batch
split = .75
#sj
train_split = np.random.choice(np.unique(sj.loc[:,'batch']),
            round(len(np.unique(sj.loc[:,'batch']))*split))
sj_train = sj[sj['batch'].isin(train_split)]
sj_test = sj.drop(sj_train.index)
#iq
train_split = np.random.choice(np.unique(iq.loc[:,'batch']),
            round(len(np.unique(iq.loc[:,'batch']))*split))
iq_train = iq[iq['batch'].isin(train_split)]
iq_test = iq.drop(iq_train.index)
#w2 sj
w2_sj_train,w2_sj_test=train_test_split(w2_sj,test_size=split)
#w2 iq
w2_iq_train,w2_iq_test=train_test_split(w2_iq,test_size=split)

#%%
#Create tensorflow graph
sj_regressor = tf.contrib.learn.DNNRegressor(feature_columns=tensors,
                                          hidden_units=[20,20],
                                          model_dir="C:/Users/schwi/Google Drive/Data Projects/Dengue Prediction/Github")

#%%
#Train the regressor NN
sj_regressor.fit(input_fn=lambda: input_fun(sj_train), steps=5000)

#%%
