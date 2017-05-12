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

#Create feature DF with 2 weeks features
w2_features <- features

#prepare new columns and names
names <- colnames(features[,5:ncol(features)-2])
for(i in 1:length(names)){
  names[i] <- paste(names[i],'prev',sep="_")
}
w2_features[,c(names)] <- NA

#Add previous week's features
for(i in 2:nrow(w2_features)){
  if(w2_features$city[i]==w2_features$city[i-1] &
     ((as.Date(w2_features$week_start_date[i])
      -as.Date(w2_features$week_start_date[i-1]))<10)){
    w2_features[i,25:44] <- w2_features[i-1,5:24]   
  }
}

#Combine features and labels
features$total_cases    <- labels$total_cases
w2_features$total_cases <- labels$total_cases

#remove rows with missing data
features              <- features[complete.cases(features),]
rownames(features)    <- NULL
w2_features           <- w2_features[complete.cases(w2_features),]
rownames(w2_features) <- NULL
#looks like we lose about 115 observations in the original feature set
#and close to 1/3 of the observations in w2_features
#This could be a problem if we don't have enought
#obvservations for our dimensionality, but we will
#examine this further later

#Break into 2 datasets by location
sj <- features[features$city=='sj',]
iq <- features[features$city=='iq',]

#List of city DFs so don't have to duplicate steps
cities <- list(sj,iq)

#Split into batches
for(j in 1:length(cities)){
  df          <- cities[[j]]
  df$batch    <- NA
  x           <- 1
  df$batch[1] <- x
  for(i in 2:nrow(df)){
    if(df$city[i]==df$city[i-1] &
       ((as.Date(df$week_start_date[i])
         -as.Date(df$week_start_date[i-1]))<10)){
      df$batch[i] <- x  
    }
    else{
      x <- x+1
      df$batch[i] <- x  
    }
  }
  if(j==1){
    sj          <- df
    cities[[j]] <- sj
  }
  if(j==2){
    iq          <- df
    cities[[j]] <- iq
  }
}


###############
## Explore the data
attributes(sj)
summary(sj)
pairs(sj[5:ncol(sj)])


###################
##Playground

#Exploring the number of continuous dates
features$check <- NA

for(i in 2:nrow(features))
{
  if(features$city[i]==features$city[i-1] &
     ((as.Date(features$week_start_date[i])
       -as.Date(features$week_start_date[i-1]))<10))
  {
    features$check[i] <- TRUE  
  }
  else
  {
    features$check[i] <- FALSE
  }
}
summary(features)
