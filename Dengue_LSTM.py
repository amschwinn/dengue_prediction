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
from keras.preprocessing import sequence as S


#set our working directory
#os.chdir(os.path.dirname('__file__'))

#%%
#Load in existing DFs exported as CSV from R

#Standard bucket datasets
iq = pd.read_csv("Data\iq_features.csv")
sj = pd.read_csv("Data\sj_features.csv")
#Non-bucket datasets
iq_nb = pd.read_csv("Data\iq_features.csv")
sj_nb = pd.read_csv("Data\sj_features.csv")
#Submission datasets
iq_sub = pd.read_csv("Data\iq_features.csv")
sj_sub = pd.read_csv("Data\sj_features.csv")

#list of DF's
cities = [sj,iq,sj_nb,iq_nb,sj_sub,iq_sub]

#Remove R cols
for i in cities:
    del i[i.columns[0]]
    
#%%
#List of feature columns
features = sj.drop(sj.columns[(len(sj.columns)-2):(len(sj.columns))],
                axis=1).drop(sj.columns[0:2],axis=1).drop(sj.columns[3],axis=1)
features = [i for i in features]
    
#%%
#Feature Normalization
for j in cities:
    for i in features:
        j[i] = (j[i] - j[i].mean())/j[i].std(ddof=0)


#%%
#Label normalization and dict of normalization variables
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
#Prepare data in correct format for Keras feed

feed = {}
#Bucketing and padding
for i in range(len(cities[:2])):
    df = cities[i]
    pre = df.loc[1,"city"]
    #Find the largets bucket size
    big_batch = np.zeros(shape=(0,0))
    for j in I.groupby(df['batch']):
        big_batch = np.append(big_batch,len(list(j[1])))    
    big_batch = int(big_batch.max())
    #Initialize arrays to fill
    in_arr = np.zeros(shape=(big_batch,len(features)))
    out_arr = np.zeros(shape=(big_batch,1))
    #Iterate through the each of the buckets
    for k in df['batch'].unique():
        in_arr2 = df.loc[(df['batch']==k)].as_matrix(features) 
        in_arr2 = np.concatenate((in_arr2, np.zeros(shape=
            (big_batch-len(in_arr2),len(features)))), axis=0)
        out_arr2 = df.loc[(df['batch']==k)].as_matrix(['total_cases']) 
        out_arr2 = np.concatenate((out_arr2, np.zeros(shape=
            (big_batch-len(out_arr2),1))), axis=0)
        if k==1:
            in_arr = np.append([in_arr],[in_arr2],axis=0)
            out_arr = np.append([out_arr],[out_arr2],axis=0)
        else:
            in_arr = np.append(in_arr,[in_arr2],axis=0)
            out_arr = np.append(out_arr,[out_arr2],axis=0)
    #Remove initial buffer arrary
    feed[(pre+"_in")] = np.delete(in_arr,0,0)
    feed[(pre+"_out")] = np.delete(out_arr,0,0)
    norm_vars[(pre+"_big_batch")] = big_batch
    
#For non-bucket sets
x = 0
for i in cities[2:]:
    pre = i.loc[1,"city"]
    #Don't need out column for submission sets
    if x < 2:
        pre = pre + '_nb'
        feed[(pre+"_out")] = i.as_matrix(['total_cases'])
    else:
        pre = pre + '_sub'
    feed[(pre+"_in")] = i.as_matrix(features)
    x += 1

#%%
#split into training and test
np.random.seed(1991)
#Specify % train/test split
split = .75
feed_split = {}
train_split = {}
#iterate through in and out arrays
for name, array in feed.items():
    #Bucket sets split
    if ('sub' not in name) & ('nb' not in name):
        #Get single training split for each city
        if name[:2] not in train_split:
            train_split[name[:2]] = np.random.choice(range(len(array)),
                        round(len(array)*split),replace=False)
        #Split training and test sets
        feed_split[(str(name)+"_train")] = array[train_split[name[:2]],:,:]
        feed_split[(str(name)+"_test")] = np.delete(array,train_split[name[:2]],0)

#%%
#Build LSTM Models
models = {}

#Data Input feed
models['sj_in'] = L.Input(shape=(norm_vars['sj_big_batch'],len(features)))
models['iq_in'] = L.Input(shape=(norm_vars['iq_big_batch'],len(features)))

#Output structure
models['sj_out'] = L.LSTM(1, return_sequences=True)(models['sj_in'])
models['iq_out'] = L.LSTM(1, return_sequences=True)(models['iq_in'])

#Create full LSTM model
models['sj_lstm'] = M.Model(input=models['sj_in'],output=models['sj_out'])
models['iq_lstm'] = M.Model(input=models['iq_in'],output=models['iq_out'])

#%%
#Train LSM Models

#Specify hyperparameters
models['sj_lstm'].compile('sgd', 'mean_squared_error')
models['iq_lstm'].compile('sgd', 'mean_squared_error')

#Fit with training set, validate with test
models['sj_lstm'].fit(feed_split['sj_in_train'], feed_split['sj_out_train'],
    epochs=15,validation_data=(feed_split['sj_in_test'], 
    feed_split['sj_out_test']))

models['iq_lstm'].fit(feed_split['iq_in_train'], feed_split['iq_out_train'],
    epochs=15,validation_data=(feed_split['iq_in_test'], 
    feed_split['iq_out_test']))

#%%
#Make predictions of submission dataset
test_feed = S.pad_sequences(feed['sj_sub_in'], maxlen=norm_vars['sj_big_batch'])
#%%
feed['sj_predictions'] = models['sj_lstm'].predict(test_feed)


#%%
#Evaluate the models
sj_score, sj_acc = models['sj_lstm'].evaluate(feed_split['sj_in_test'],
                         feed_split['sj_out_test'])

iq_score, iq_acc = models['iq_lstm'].evaluate(feed_split['iq_in_test'],
                         feed_split['iq_out_test'])

print("sj_score:"+str(sj_score))
print("sj_acc:"+str(sj_acc))
print("iq_score:"+str(iq_score))
print("iq_acc:"+str(iq_acc))
