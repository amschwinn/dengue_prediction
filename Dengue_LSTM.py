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
import math as ma
from keras.preprocessing import sequence as S


#set our working directory
#os.chdir(os.path.dirname('__file__'))

#%%
#Load in existing DFs exported as CSV from R

#Standard bucket datasets
iq = pd.read_csv("Data\iq_features.csv")
sj = pd.read_csv("Data\sj_features.csv")
#Non-bucket datasets
iq_nb = pd.read_csv("Data\iq_nb.csv")
sj_nb = pd.read_csv("Data\sj_nb.csv")
#Submission datasets
iq_sub = pd.read_csv("Data\iq_sub.csv")
sj_sub = pd.read_csv("Data\sj_sub.csv")

#list of DF's
cities = [sj,iq,sj_nb,iq_nb,sj_sub,iq_sub]
submiss = [sj_sub,iq_sub]


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
for i in range(2):
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
#Evaluate the models
sj_score, sj_acc = models['sj_lstm'].evaluate(feed_split['sj_in_test'],
                         feed_split['sj_out_test'])

iq_score, iq_acc = models['iq_lstm'].evaluate(feed_split['iq_in_test'],
                         feed_split['iq_out_test'])

print("sj_score:"+str(sj_score))
print("sj_acc:"+str(sj_acc))
print("iq_score:"+str(iq_score))
print("iq_acc:"+str(iq_acc))
#%%
#Make predictions of submission dataset

for i in ['sj','iq']:
    #Prepare submission feed to send through model
    #Must be in proper array shape
    feed_split[(i+'_sub_in')] = np.zeros(shape=(norm_vars[(i+'_big_batch')],
        feed[(i+'_sub_in')].shape[1]))
    
    #Split the full array in subarrays of the size of buckets
    for low in range(0,len(feed[(i+'_sub_in')]),norm_vars[(i+'_big_batch')]):
        #Pad final array to correct shape
        if low + norm_vars[(i+'_big_batch')] > len(feed[(i+'_sub_in')]):
            high = len(feed[(i+'_sub_in')])
            feed_split[(i+'_sub_in')] = np.append(feed_split[(i+'_sub_in')],
                [np.append(feed[(i+'_sub_in')][low:high,:],
                np.zeros(shape=((norm_vars[(i+'_big_batch')]-(high-low)),
                feed[(i+'_sub_in')].shape[1])),axis=0)],axis=0)
        #Need to alter format during first iteration
        elif low < 17:
            high = low + norm_vars[(i+'_big_batch')]
            feed_split[(i+'_sub_in')] = np.append([feed_split[(i+'_sub_in')]],
                [feed[(i+'_sub_in')][low:high,:]],axis=0)    
        #For arrays that don't need padding
        else:
            high = low + norm_vars[(i+'_big_batch')]
            feed_split[(i+'_sub_in')] = np.append(feed_split[(i+'_sub_in')],
                [feed[(i+'_sub_in')][low:high,:]],axis=0)
    #Remove first buffer array
    feed_split[(i+'_sub_in')] = np.delete(feed_split[(i+'_sub_in')],0,0)
    
    #Predict using the model
    feed[(i+'_predictions')] = models[(i+'_lstm')].predict(feed_split[(i+
        '_sub_in')])
    
    #Flatten from 3D array to 2D Dataframe
    feed[(i+'_predictions')] = pd.Panel(feed[(i+'_predictions')]).swapaxes(0,
        2).to_frame().reset_index().iloc[:len(feed[(i+'_sub_in')]),2]
    
    #De-normalize the predictions
    feed[(i+'_predictions')] = round((feed[(i+'_predictions')]*norm_vars[(
        i+'_std')])+norm_vars[(i+'_mean')])
    #Specify col name
    #feed[(i+'_predictions')] = feed[(i+'_predictions')].rename('total_cases')
    feed[(i+'_predictions')] = feed[(i+'_predictions')].to_frame('total_cases')
#%%
#Combine to proper output
for i in range(len(submiss)):
    submiss[i] = pd.merge(submiss[i].loc[:,['city','year','weekofyear']],
           feed[('sj_predictions')],left_index=True,right_index=True)
    
#Combine cities
tot_sub = pd.concat(submiss)

#Total cases as int
tot_sub['total_cases'] = tot_sub['total_cases'].apply(np.int64)

#Output submission to CSV
tot_sub.to_csv("Data/LSTM_Bucket_Submission_1.csv",index=False)

