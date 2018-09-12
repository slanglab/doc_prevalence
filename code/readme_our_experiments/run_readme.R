#!/usr/bin/env Rscript
library(ReadMe)

args = commandArgs(trailingOnly=TRUE)
setting = args[1] #"nat" or "prop1"
trial = args[2]
testgroup = args[3] #1 thru 500

#YOU WILL NEED TO CHANGE THIS MANUALLY! 
HOME_PATH = '/home/kkeith/docprop/code/readme_our_experiments/'

TRAIN_PATH = paste(HOME_PATH, 'train/', setting, '/trial', trial, '/', sep="")
trainX = read.csv(paste(TRAIN_PATH, 'trainX.csv', sep=""),sep=",",header=F)
trainY = read.csv(paste(TRAIN_PATH, 'trainY.csv', sep=""),sep=",",header=F)
stopifnot(nrow(trainX) == 2000)
stopifnot(nrow(trainX) == 2000)

TEST_PATH = paste(HOME_PATH, 'test/', setting, '/trial', trial, '/', sep="")
testX = read.csv(paste(TEST_PATH, 'group', testgroup, 'X.csv', sep=""),sep=",",header=F)
testY = read.csv(paste(TEST_PATH, 'group', testgroup, 'Y.csv', sep=""),sep=",",header=F)
stopifnot(nrow(testX) == nrow(testY))

#make sure vocab size is the same 
stopifnot(ncol(trainX) == ncol(testX))

#change from counts to indicators
trainX[] <- lapply(trainX, function(x) ifelse(x>1, as.integer(1), x))
testX[] <- lapply(testX, function(x) ifelse(x>1, as.integer(1), x))

sttxt <- colnames(trainX)[1]
fntxt <- colnames(trainX)[ncol(trainX)]

# Prepare to call the ReadMe package's preprocess() and readme() functions.
# (readme() in turn calls VA's va()).

#add headers that are needed for input into readme 
#TRAINING
trainX$FILENAME <- factor(rep(1, nrow(trainX))) #will just make "1" the filenames, this just gets deleted later on anyways 
trainX$TRUTH <- trainY$V1
ncolms <- ncol(trainX)

#need to reorder the columns so that they match the original program with filename and truth as first and second 
trainX <- trainX[c(ncolms-1, ncolms, 1:(ncolms-2))]

#TESTING
testX$FILENAME <- factor(rep(1, nrow(testX))) #blah data for filename (this gets deleted anyways)
testX$TRUTH <- testY$V1

#reorder colmns so that that match the original program
testX <- testX[c(ncolms-1, ncolms, 1:(ncolms-2))]

#add training, testing, codes 
trainX$TRAININGSET <- factor(rep(1, nrow(trainX)))
testX$TRAININGSET<- factor(rep(0, nrow(testX)))

# We want to use the same default settings that the ReadMe software package
# uses.  These default settings are below, copied from the bottom of
# undergrad() in prototype.R.
# See: https://github.com/iqss-research/ReadMeV1/blob/master/R/prototype.R
ret <- list()
ret$trainingset <- trainX
ret$testset <- testX

cnames <- colnames(trainX)
ncols <- length(cnames)
formula <- paste(sttxt,"+...+",fntxt,"~TRUTH",sep="")
formula <- as.formula(formula)

ret$formula <- formula
ret$features <- 15 
ret$n.subset <- 300
ret$prob.wt <- 1
ret$boot.se <- FALSE
ret$nboot = 300
ret$printit = FALSE

#run thru readme package 
pp <-  preprocess(ret)
vout1 <- readme(pp)

#get MAE
est <- vout1$est.CSMF[2] 
true <- vout1$true.CSMF[2]

#AE, bias, est, true 
ae <- c(abs(est-true), est-true, est, true) 

#write mae results to csv for aggregation 
save_file = paste(HOME_PATH, 'results/', setting, '/trial', trial, '/group', testgroup, '.csv', sep='')
print(save_file)
write.table(ae, file=save_file, sep=",",  col.names=FALSE, row.names=FALSE)
