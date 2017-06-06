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
import os
import numpy as np
import keras.layers as L
import keras.models as M
import itertools as I


#set our working directory
#os.chdir(os.path.dirname('__file__'))

#%%
#Load in existing DFs exported as CSV from R
iq = pd.read_csv("iq_features.csv")
sj = pd.read_csv("sj_features.csv")

#list of DF's
cities = [sj,iq]

#Remove R cols
for i in cities:
    del i[i.columns[0]]
    
#%%
#List of feature columns
features = sj.drop(sj.columns[(len(sj.columns)-2):(len(sj.columns))],
                   axis=1).drop(sj.columns[0:4],axis=1)
features = [i for i in features]
    
#%%
#Feature Normalization
for i in features:
    sj[i] = (sj[i] - sj[i].mean())/sj[i].std(ddof=0)
    iq[i] = (iq[i] - iq[i].mean())/iq[i].std(ddof=0) 

#%%
#Label normalization
norm_vars = {}
for i in range(len(cities)):
    df = cities[i]
    pre = df.loc[1,"city"]
    mean = df["total_cases"].mean()
    std = df["total_cases"].std(ddof=0)
    cities[i]["total_cases"] = (df["total_cases"] - mean)/std
    norm_vars[(pre+"_mean")] = mean
    norm_vars[(pre+"_std")] = std

#%%
#Prepare for bucketing and padding

#Find the largets bucket size
big_batch = np.zeros(shape=(0,0))
for i in I.groupby(sj['batch']):
    big_batch = np.append(big_batch,len(list(i[1])))    
big_batch = int(big_batch.max())
#Initialize arrays to fill
in_arr = np.zeros(shape=(big_batch,len(features)))
out_arr = np.zeros(shape=(big_batch,1))
#Iterate through the each of the buckets
for i in sj['batch'].unique():
    in_arr2 = sj.loc[(sj['batch']==i)].as_matrix(features) 
    in_arr2 = np.concatenate((in_arr2, np.zeros(shape=(big_batch-len(in_arr2),
        len(features)))), axis=0)
    out_arr2 = sj.loc[(sj['batch']==i)].as_matrix(['total_cases']) 
    out_arr2 = np.concatenate((out_arr2, np.zeros(shape=
        (big_batch-len(out_arr2),1))), axis=0)
    if i==1:
        in_arr = np.append([in_arr],[in_arr2],axis=0)
        out_arr = np.append([out_arr],[out_arr2],axis=0)
    else:
        in_arr = np.append(in_arr,[in_arr2],axis=0)
        out_arr = np.append(out_arr,[out_arr2],axis=0)
#Remove initial buffer arrary
in_arr = np.delete(in_arr,0,0)
out_arr = np.delete(out_arr,0,0)

#%%
sj_in = in_arr
sj_out = out_arr

#%%
#split into training and test
np.random.seed(1991)
#Split original by batch
split = .75
#sj
train_split = np.random.choice(range(len(sj_in)),
            round(len(sj_in)*split),replace=False)
sj_in_train = sj_in[train_split,:,:]
sj_out_train = sj_out[train_split,:,:]
sj_in_test = np.delete(sj_in,train_split,0)
sj_out_test = np.delete(sj_out,train_split,0)

#%%
sj_in = L.Input(shape=(big_batch,len(features)))

sj_out = L.LSTM(1, return_sequences=True)(sj_in)

sj_lstm = M.Model(input=sj_in,output=sj_out)

sj_lstm.compile('sgd', 'mean_squared_error')

sj_lstm.fit(sj_in_train, sj_out_train,epochs=15,
          validation_data=(sj_in_test, sj_out_test))

#score, acc = sj_lstm.evaluate(sj_in_test, sj_out_test)

