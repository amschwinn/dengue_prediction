##############################################
#By: Austin Schwinn
#Date: May 5, 2017
#Subject: Integrating tensorflow in R
#Based on https://github.com/rstudio/tensorflow
##############################################

#install.packages('devtools')
#install.packages('reticulate')
#Separetly download/install RBuildTools & restart

#Before using this code, create a virtual environment
#in anaconda following tensorflow's Python installation
#instructions. Make sure to specify Python 3.5.2 when
#creating the virtual environment

library(reticulate)
library(devtools)

#Use dev_mode to create and install tensorflow package
dev_mode(on=T)
devtools::install_github("rstudio/tensorflow")

#Supported by anaconda virtual environment of tensorflow
use_condaenv('tensorflow', conda = "auto", required = TRUE)
library(tensorflow)
dev_mode(on=F)

#Veryifying the installation
library(tensorflow)
sess = tf$Session()
hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)
