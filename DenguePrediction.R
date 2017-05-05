##############################################
#By: Austin Schwinn
#Date: April 26, 2017
#Subject: Predicting mosquito cause dengue
#outbreaks in Puerto Rico and Peru
##############################################
#install.packages('rstudioapi')
library(rstudioapi)

#Before moving forward, please open, read, and run
#the TensorFlow_R_Integration file
library(tensorflow)

#Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Load our datasets
features    <- read.csv('dengue_features_train.csv')
labels      <- read.csv('dengue_labels_train.csv')
submission  <- read.csv('dengue_features_test.csv')

#prepare new columns and names
names <- colnames(features[,5:ncol(features)-2])
for(i in 1:length(names))
{
  names[i] <- paste(names[i],'prev',sep="_")
}
features[,c(names)] <- NA

#Add previous week's features
for(i in 2:nrow(features))
{
  if(features$city[i]==features$city[i-1] &
     ((as.Date(features$week_start_date[i])
      -as.Date(features$week_start_date[i-1]))<10))
  {
    features[i,25:44] <- features[i-1,5:24]   
  }
}

#Combine features and labels
features$total_cases <- labels$total_cases

#remove rows with missing data
features <- features[complete.cases(features),]
rownames(features) <- NULL
#looks like we lose about 1/3 of observations
#This could be a problem if we don't have enought
#obvservations for our dimensionality, but we will
#examine this further later

#Break into 2 datasets by location
sj <- features[features$city=='sj',]
iq <- features[features$city=='iq',]

