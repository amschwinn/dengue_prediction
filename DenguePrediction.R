f##############################################
#By: Austin Schwinn
#Date: April 26, 2017
#Subject: Predicting mosquito cause dengue
#outbreaks in Puerto Rico and Peru
##############################################
#install.packages('rstudioapi')
#install.packages('kerasR')
#install.packages('neuralnet')
library(rstudioapi)
library(kerasR)
library(neuralnet)

#Before moving forward, please open, read, and run
#the TensorFlow_R_Integration file
library(tensorflow)

#Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Load our datasets
features    <- read.csv('dengue_features_train.csv')
labels      <- read.csv('dengue_labels_train.csv')
submission  <- read.csv('dengue_features_test.csv')


#Combine features and labels
features$total_cases    <- labels$total_cases

#Create feature DF with 2 weeks features
w2_features <- features

#remove rows with missing data
features              <- features[complete.cases(features),]
rownames(features)    <- NULL
#looks like we lose about 115 observations in the original feature set

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


##############
## 2 weeks of features

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
w2_features$total_cases <- labels$total_cases

#remove rows with missing data
w2_features           <- w2_features[complete.cases(w2_features),]
rownames(w2_features) <- NULL


#Break into 2 datasets by location
w2_sj <- w2_features[w2_features$city=='sj',]
w2_iq <- w2_features[w2_features$city=='iq',]


# Export working DF's to use in Tensorflow in Python
write.csv(iq, "iq_features")
write.csv(sj, "sj_features")
write.csv(w2_iq, "w2_iq_features")
write.csv(w2_sj, "w2_sj_features")




######################
## Modelinng

#Normalize the data
for(i in 1:length(cities)){
  df      <- cities[[i]]
  df_max  <- apply(df[,5:(ncol(df)-1)],2,max)
  df_min  <- apply(df[,5:(ncol(df)-1)],2,min)
  df_norm <- as.data.frame(cbind(df[,1:4],scale(df[,5:(ncol(df)-1)], center=df_min,
                                        scale=df_max-df_min),df[,ncol(df)]))
  names(df_norm) <- names(df)
  if(i==1){
    sj_norm <- df_norm
  }
  if(i==2){
    iq_norm <- df_norm
  }
}

#Split batches into train and test
set.seed(1991)
split_size  <- .75
sj_index    <- sample(seq_len(max(sj$batch)), size=(split_size*max(sj$batch)))
iq_index    <- sample(seq_len(max(iq$batch)), size=(split_size*max(iq$batch)))
sj_train    <- sj_norm[sj$batch %in% sj_index,-c(1:4)]
sj_test     <- sj_norm[!(sj$batch %in% sj_index),-c(1:4)]
iq_train    <- iq_norm[iq$batch %in% iq_index,-c(1:4)]
iq_test     <- iq_norm[!(iq$batch %in% iq_index),-c(1:4)]

###############
##Standard Neural Net
#Uses the following tutorial:
#https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/

#Create target equation
feat_names  <- names(sj_train[1:(ncol(sj_train)-2)])
equation    <- as.formula(paste("total_cases~",paste(feat_names, collapse="+")))

#Neural net regression
sj_nn <- neuralnet(equation,data=sj_train[1:(ncol(sj_train)-1)],
                hidden=c(12,7,4,3),linear.output=TRUE)
iq_nn <- neuralnet(equation,data=iq_train[1:(ncol(sj_train)-1)],
                   hidden=c(12,7,4,3),linear.output=TRUE)

#Predict from fitted NN model
pr.sj_nn      <- compute(sj_nn,sj_test[,1:(ncol(sj_test)-2)])
sj_nn_result  <- pr.sj_nn$net.result*(max(sj$total_cases)
                                       -min(sj$total_cases))+min(sj$total_cases)
sj_test_pred  <- sj_test$total_cases*(max(sj$total_cases)
                                      -min(sj$total_cases))+min(sj$total_cases)

#MSE of NN model
sum((sj_test_pred-sj_nn_result)^2)/nrow(sj_test)


##############
## Tensorflow linear model w/ deep learning

#Create feature columns for the model
feat_col <- names(features[,5:ncol(features)])

for(i in 1:length(feat_col)){
  assign(feat_col[i],tf$contrib$layers$real_valued_column(feat_col[i]))
}



##############
## Further development
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
w2_features$total_cases <- labels$total_cases

#remove rows with missing data
w2_features           <- w2_features[complete.cases(w2_features),]
rownames(w2_features) <- NULL

