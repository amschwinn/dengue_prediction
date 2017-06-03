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
import pylab as pl
import time
from sklearn.model_selection import train_test_split
from decimal import *

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
#List of feature columns
features = sj.drop(sj.columns[(len(sj.columns)-2):(len(sj.columns))],
                   axis=1).drop(sj.columns[0:4],axis=1)
features = [i for i in features]

w2_features = w2_sj.drop(w2_sj.columns[(len(w2_sj.columns)-1)],
                         axis=1).drop(w2_sj.columns[0:4],axis=1)
w2_features = [i for i in w2_features]
    
    
#%%
#Feature Normalization
for i in features:
    sj[i] = (sj[i] - sj[i].mean())/sj[i].std(ddof=0)
    iq[i] = (iq[i] - iq[i].mean())/iq[i].std(ddof=0) 
    
for i in w2_features:
    w2_sj[i] = (w2_sj[i] - w2_sj[i].mean())/w2_sj[i].std(ddof=0)
    w2_iq[i] = (w2_iq[i] - w2_iq[i].mean())/w2_iq[i].std(ddof=0)

#%%
#Label normalization
norm_vars = {}
for i in range(len(cities)):
    df = cities[i]
    if i < 2:
        pre = df.loc[1,"city"]
    else:
        pre = "w2_"+df.loc[1,"city"]
    mean = df["total_cases"].mean()
    std = df["total_cases"].std(ddof=0)
    cities[i]["total_cases"] = (df["total_cases"] - mean)/std
    norm_vars[(pre+"_mean")] = mean
    norm_vars[(pre+"_std")] = std

 
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
#Create tensors for each feature column
w1_feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in features]

#Now the same for the 2 week look back
w2_feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in w2_features]

#Prediction value
label = ["total_cases"]    

#%%
#Create input functions
def input_fun(data_set):
    w1_feature_cols = {k: tf.constant(data_set[k].values)
        for k in features}
    w1_labels = tf.constant(data_set[label].values)
    return w1_feature_cols, w1_labels

#same for w2
def w2_input_fun(data_set):
    w2_feature_cols = {k: tf.constant(data_set[k].values)
        for k in w2_features}
    w2_labels = tf.constant(data_set[label].values)
    return w2_feature_cols, w2_labels
#%%
#Test for optimal hyperparameters
'''
#Track how long the loop takes
start = time.time()
#Create DF to store results of tests
results = pd.DataFrame(columns=["lay1","learn_rate","l1","l2","sj_loss","iq_loss"])
ind = 0

#Testing loop
results = pd.DataFrame(columns=["lay1","l1","l2","sj_loss","iq_loss"])
ind = 0

#Testing loop
for lay1 in range(1,41):
    for l1 in pl.frange(0.0,1.0,0.1):
        for l2 in pl.frange(0.0,1.0,0.1):                
            #Insert hyperparameters
            w2_sj_regressor = tf.contrib.learn.DNNRegressor(
                feature_columns=w2_feature_cols, hidden_units=[lay1],
                optimizer=tf.train.FtrlOptimizer(learning_rate=.1,
                l1_regularization_strength=l1,
                l2_regularization_strength=l2),
                model_dir=
                ("C:/Users/schwinnter/Documents/models/sj/"+str(ind)))
            w2_iq_regressor = tf.contrib.learn.DNNRegressor(
                feature_columns=w2_feature_cols, hidden_units=[lay1],
                optimizer=tf.train.FtrlOptimizer(learning_rate=.1,
                l1_regularization_strength=l1,
                l2_regularization_strength=l2),
                model_dir=
                ("C:/Users/schwinnter/Documents/models/iq/"+str(ind))) 
            #Fit the models
            w2_sj_regressor.fit(input_fn=lambda: w2_input_fun(w2_sj_train),
                steps=5000)
            w2_iq_regressor.fit(input_fn=lambda: w2_input_fun(w2_iq_train), 
                steps=5000)
            #Evalute loss with test
            w2_sj_ev = w2_sj_regressor.evaluate(input_fn=lambda: 
                w2_input_fun(w2_sj_test), steps=1)
            sj_loss = w2_sj_ev["loss"]
            
            w2_iq_ev = w2_iq_regressor.evaluate(input_fn=lambda: 
                w2_input_fun(w2_iq_test), steps=1)
            iq_loss = w2_iq_ev["loss"]
            #Update results DF and iterators
            results.loc[ind,:] = [lay1,l1,l2,sj_loss,iq_loss]
            ind += 1
            #output progress
            results.to_csv("test_results.csv")
#End timer and display time it took
end = time.time()
print(end - start)

#Export the results DF to CSV
results.to_csv("test_results.csv")
'''
#%%
#Read in test results
results = pd.read_csv("test_results.csv")

#Select the hyperparameters with the smallest loss
sj_min = results[(results['sj_loss']<=(results['sj_loss'].min())+.01)]

#%%
results['sj_loss'].min()

#%%
#Create tensorflow graph
sj_regressor = tf.contrib.learn.DNNRegressor(feature_columns=w1_feature_cols,
    hidden_units=[20,20], optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
    model_dir=
    "C:/Users/schwi/Google Drive/Data Projects/Dengue Prediction/models/sj_dnn_reg")
    
iq_regressor = tf.contrib.learn.DNNRegressor(feature_columns=w1_feature_cols,
    hidden_units=[20,20], optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
    model_dir=
    "C:/Users/schwi/Google Drive/Data Projects/Dengue Prediction/models/iq_dnn_reg")

w2_sj_regressor = tf.contrib.learn.DNNRegressor(feature_columns=w2_feature_cols,
    hidden_units=[40,40], optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
    model_dir=
    "C:/Users/schwi/Google Drive/Data Projects/Dengue Prediction/models/w2_sj_dnn_reg")

w2_iq_regressor = tf.contrib.learn.DNNRegressor(feature_columns=w2_feature_cols,
    hidden_units=[40,40], optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
    model_dir=
    "C:/Users/schwi/Google Drive/Data Projects/Dengue Prediction/models/w2_iq_dnn_reg")
#%%
#Train the regressor NN
sj_regressor.fit(input_fn=lambda: input_fun(sj_train), steps=5000)

iq_regressor.fit(input_fn=lambda: input_fun(iq_train), steps=5000)

#W2
w2_sj_regressor.fit(input_fn=lambda: w2_input_fun(w2_sj_train), steps=5000)

w2_iq_regressor.fit(input_fn=lambda: w2_input_fun(w2_iq_train), steps=5000)

#%%
#Evaluate the models
sj_ev = sj_regressor.evaluate(input_fn=lambda: input_fun(sj_test), steps=1)
sj_loss_score = sj_ev["loss"]

iq_ev = iq_regressor.evaluate(input_fn=lambda: input_fun(iq_test), steps=1)
iq_loss_score = iq_ev["loss"]

w2_sj_ev = w2_sj_regressor.evaluate(input_fn=lambda: w2_input_fun(w2_sj_test), steps=1)
w2_sj_loss_score = w2_sj_ev["loss"]

w2_iq_ev = w2_iq_regressor.evaluate(input_fn=lambda: w2_input_fun(w2_iq_test), steps=1)
w2_iq_loss_score = w2_iq_ev["loss"]


#See the test loss results
print("SJ Loss: {0:f}".format(sj_loss_score))
print("iq Loss: {0:f}".format(iq_loss_score))
print("W2 SJ Loss: {0:f}".format(w2_sj_loss_score))
print("W2 iq Loss: {0:f}".format(w2_iq_loss_score))
#%%
#Predict from the model
sj_predict = sj_regressor.predict(input_fn=lambda: input_fun(sj_test))
# .predict() returns an iterator; convert to a list and print predictions
sj_predictions = list(itertools.islice(sj_predict, 6))
print ("Predictions: {}".format(str(sj_predictions)))


#%%
#Test for optimal hyperparameters

#Track how long the loop takes
start = time.time()
#Create DF to store results of tests
results = pd.DataFrame(columns=["lay1","learn_rate","l1","l2","sj_loss","iq_loss"])
ind = 0

#Testing loop
for lay1 in range(1,41):
    for learn_rate in pl.frange(0.1,1.0,0.1):
        for l1 in pl.frange(0.0,1.0,0.1):
            for l2 in pl.frange(0.0,1.0,0.1):                
                #Insert hyperparameters
                w2_sj_regressor = tf.contrib.learn.DNNRegressor(
                    feature_columns=w2_feature_cols, hidden_units=[lay1],
                    optimizer=tf.train.FtrlOptimizer(learning_rate=learn_rate,
                    l1_regularization_strength=l1,
                    l2_regularization_strength=l2),
                    model_dir=
                    ("C:/Users/schwi/Google Drive/Data Projects/Dengue " + 
                     "Prediction/models/w2_sj_dnn_reg/"+str(ind)))
                w2_iq_regressor = tf.contrib.learn.DNNRegressor(
                    feature_columns=w2_feature_cols, hidden_units=[lay1],
                    optimizer=tf.train.FtrlOptimizer(learning_rate=learn_rate,
                    l1_regularization_strength=l1,
                    l2_regularization_strength=l2),
                    model_dir=
                    ("C:/Users/schwi/Google Drive/Data Projects/Dengue " + 
                     "Prediction/models/w2_iq_dnn_reg/"+str(ind))) 
                #Fit the models
                w2_sj_regressor.fit(input_fn=lambda: w2_input_fun(w2_sj_train),
                    steps=5000)
                w2_iq_regressor.fit(input_fn=lambda: w2_input_fun(w2_iq_train), 
                    steps=5000)
                #Evalute loss with test
                w2_sj_ev = w2_sj_regressor.evaluate(input_fn=lambda: 
                    w2_input_fun(w2_sj_test), steps=1)
                sj_loss = w2_sj_ev["loss"]
                
                w2_iq_ev = w2_iq_regressor.evaluate(input_fn=lambda: 
                    w2_input_fun(w2_iq_test), steps=1)
                iq_loss = w2_iq_ev["loss"]
                #Update results DF and iterators
                results.loc[ind,:] = [lay1,learn_rate,l1,l2,sj_loss,iq_loss]
                ind += 1
                #Display progress
                print(ind/48400)
#End timer and display time it took
end = time.time()
print(end - start)
