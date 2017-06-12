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
import keras.optimizers as O
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

#Specify size of batch sizes
norm_vars['sj_big_batch'] = 10
norm_vars['iq_big_batch'] = 10

feed = {}
#Bucketing and padding
for i in range(len(cities[2:4])):
    df = cities[i]
    pre = df.loc[1,"city"]
    #Initialize arrays to fill
    feed[(pre+'_in')] = np.zeros(shape=(norm_vars[(pre+'_big_batch')],
    len(features)))
    feed[(pre+'_out')] = np.zeros(shape=(norm_vars[(pre+'_big_batch')],1))
    #Split the full array in subarrays of the size of buckets
    for low in range(0,len(df)):
        #Pad final array to correct shape
        if low + norm_vars[(pre+'_big_batch')] > len(df):
            high = len(df)
            feed[(pre+'_in')] = np.append(feed[(pre+'_in')],
                [np.append(df.loc[low:high,features].as_matrix(),
                np.zeros(shape=((norm_vars[(pre+'_big_batch')]-(high-low)),
                len(features))),axis=0)],axis=0)
            feed[(pre+'_out')] = np.append(feed[(pre+'_out')],
                [np.append(df.loc[low:high,['total_cases']].as_matrix(),
                np.zeros(shape=((norm_vars[(pre+'_big_batch')]-(high-low)),1)),
                axis=0)],axis=0)
        #Need to alter format during first iteration
        elif low == 0:
            high = low + norm_vars[(pre+'_big_batch')]-1
            feed[(pre+'_in')] = np.append([feed[(pre+'_in')]],
                [df.loc[low:high,features].as_matrix()],axis=0) 
            feed[(pre+'_out')] = np.append([feed[(pre+'_out')]],
                [df.loc[low:high,['total_cases']].as_matrix()],axis=0)
        #For arrays that don't need padding
        else:
            high = low + norm_vars[(pre+'_big_batch')]-1
            feed[(pre+'_in')] = np.append(feed[(pre+'_in')],
                [df.loc[low:high,features].as_matrix()],axis=0)
            feed[(pre+'_out')] = np.append(feed[(pre+'_out')],
                [df.loc[low:high,['total_cases']].as_matrix()],axis=0)
    #Remove first array that was used as buffer
    feed[(pre+'_in')] = np.delete(feed[(pre+'_in')],0,0)
    feed[(pre+'_out')] = np.delete(feed[(pre+'_out')],0,0)
    
#For non-bucket sets
for i in cities[4:]:
    pre = i.loc[1,"city"]
    feed[(pre+"_sub_in")] = i.as_matrix(features)

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
    if ('sub' not in name):
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

#Optimizer
rms = O.RMSprop()

#Add hyperparameters
models['sj_lstm'].compile(rms, 'mean_squared_error')
models['iq_lstm'].compile(rms, 'mean_squared_error')

#Fit with training set, validate with test
models['sj_lstm'].fit(feed_split['sj_in_train'], feed_split['sj_out_train'],
    epochs=100,validation_data=(feed_split['sj_in_test'], 
    feed_split['sj_out_test']))

models['iq_lstm'].fit(feed_split['iq_in_train'], feed_split['iq_out_train'],
    epochs=100,validation_data=(feed_split['iq_in_test'], 
    feed_split['iq_out_test']))

#%%
#Evaluate the models
sj_score = models['sj_lstm'].evaluate(feed_split['sj_in_test'],
                         feed_split['sj_out_test'])
iq_score = models['iq_lstm'].evaluate(feed_split['iq_in_test'],
                         feed_split['iq_out_test'])

print("sj_score:"+str(sj_score))
print("iq_score:"+str(iq_score))
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
        elif low < norm_vars[(i+'_big_batch')]:
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
submiss = [pd.read_csv("Data\sj_sub.csv"),pd.read_csv("Data\iq_sub.csv")]

for i in range(len(submiss)):
    pre = submiss[i].loc[0,'city']
    submiss[i] = pd.merge(submiss[i].loc[:,['city','year','weekofyear']],
           feed[(pre+'_predictions')],left_index=True,right_index=True)
    
#Combine cities
tot_sub = pd.concat(submiss)

#Total cases as int
tot_sub['total_cases'] = tot_sub['total_cases'].apply(np.int64)

#Output submission to CSV
tot_sub.to_csv("Data/LSTM_NonOmit_Submission_2.csv",index=False)

