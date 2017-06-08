f##############################################
#By: Austin Schwinn
#Date: April 26, 2017
#Subject: Predicting mosquito cause dengue
#outbreaks in Puerto Rico and Peru
##############################################
#install.packages('rstudioapi')
library(rstudioapi)

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

#backup before removing NA's
features2 <- features

#remove rows with missing data
features              <- features[complete.cases(features),]
rownames(features)    <- NULL
#looks like we lose about 115 observations in the original feature set

#Break into 2 datasets by location
sj      <- features[features$city=='sj',]
iq      <- features[features$city=='iq',]

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
feat_names <- colnames(w2_features[,5:(ncol(w2_features)-1)])
for(i in 1:length(feat_names)){
  feat_names[i] <- paste(feat_names[i],'prev',sep="_")
}
w2_features[,c(feat_names)] <- NA

#move total cases column to the end
tot_index <- grep("total_cases", names(w2_features))
w2_features <- w2_features[,c((1:ncol(w2_features))[-tot_index],tot_index)]

#Add previous week's features
for(i in 2:nrow(w2_features)){
  if(w2_features$city[i]==w2_features$city[i-1] &
     ((as.Date(w2_features$week_start_date[i])
       -as.Date(w2_features$week_start_date[i-1]))<10)){
    w2_features[i,25:44] <- w2_features[i-1,5:24]   
  }
}

#remove rows with missing data
w2_features           <- w2_features[complete.cases(w2_features),]
rownames(w2_features) <- NULL


#Break into 2 datasets by location
w2_sj <- w2_features[w2_features$city=='sj',]
w2_iq <- w2_features[w2_features$city=='iq',]

##############
## 3 weeks of features

#prepare new columns and names
feat_names <- colnames(features[,5:(ncol(features)-1)])
feat_names <- c(feat_names,paste(feat_names,"2",sep="_"))
feat_names <- paste(feat_names,"prev",sep="_")
w3_features <- features
w3_features[,c(feat_names)] <- NA

#move total cases column to the end
tot_index <- grep("total_cases", names(w3_features))
w3_features <- w3_features[,c((1:ncol(w3_features))[-tot_index],tot_index)]

#Add previous week's features
for(i in 3:nrow(w3_features)){
  if(w3_features$city[i]==w3_features$city[i-1] &
     w3_features$city[i]==w3_features$city[i-2] &
     ((as.Date(w3_features$week_start_date[i])
       -as.Date(w3_features$week_start_date[i-1]))<10) &
    ((as.Date(w3_features$week_start_date[i-1])
      -as.Date(w3_features$week_start_date[i-2]))<10)){
    w3_features[i,25:44] <- w3_features[i-1,5:24] 
    w3_features[i,45:64] <- w3_features[i-2,5:24]
  }
}

#remove rows with missing data
w3_features           <- w3_features[complete.cases(w3_features),]
rownames(w3_features) <- NULL


#Break into 2 datasets by location
w3_sj <- w3_features[w3_features$city=='sj',]
w3_iq <- w3_features[w3_features$city=='iq',]


# Export working DF's to use in Tensorflow in Python
write.csv(iq, "Data/iq_features.csv")
write.csv(sj, "Data/sj_features.csv")
write.csv(w2_iq, "Data/w2_iq_features.csv")
write.csv(w2_sj, "Data/w2_sj_features.csv")
write.csv(w3_iq, "Data/w3_iq_features.csv")
write.csv(w3_sj, "Data/w3_sj_features.csv")


##############
## Prepare Prediction Set

# Do not want to remove any observations since
# competition is based on total mean loss
# so will have to predict on every single observation.
# Instead of exluding NA's, will fill with feature avg.
# Will also make single batch feature sets with means
# instead of omitting NAs to see if it trains the model
# any better than splitting into buckets

#Split by cities
sj_nb   <- features2[features2$city=='sj',]
iq_nb   <- features2[features2$city=='iq',]
sj_sub  <- submission[submission$city=='sj',]
iq_sub  <- submission[submission$city=='iq',]

#List to iterate through
tot_obs   <- list(sj_nb,iq_nb,sj_sub,iq_sub)
tot_names <- list('sj_nb','iq_nb','sj_sub','iq_sub')

#Replace NA vals with col means
for(j in 1:length(tot_obs)){
  df  <- tot_obs[[j]]
  #Iterate through cols and use mean for NAs
  for(i in 1:ncol(df)){
    df[is.na(df[,i]), i] <- mean(df[,i], na.rm = TRUE)
  }
  #Rewrite back to list objects
  tot_obs[[j]] <- df
  #Write DF to CSV to read into Python model
  write.csv(df,paste("Data/",paste(tot_names[[j]],".csv",sep=""),sep=""))
}


