library(tidyverse)
library(lubridate)
library(reshape)
library(gamlr)
library(glmnet)
library(pROC)
library(tm)
library(xgboost)
library(text2vec)
library(data.table)

train = read.csv('train.csv', stringsAsFactors = FALSE)
test = read.csv('test.csv', stringsAsFactors = FALSE)

train.X <- as.matrix(train[, -(1:3)])
train.y <- as.numeric(train$sentiment)
test.X <- as.matrix(test[, -c(1,2)])

set.seed(998)

cv_model <- cv.glmnet(train.X, train.y, family = 'binomial', alpha=0.3, 
                      standardize = FALSE, type.measure = 'auc', nfolds = 5)
lambda.best <- cv_model$lambda.min
final_model <- glmnet(train.X, train.y, family = 'binomial', alpha=0.3, 
                      lambda = lambda.best, standardize = FALSE)

y.pred <- predict(final_model, newx = test.X, type = 'response')

submission <- data.table(
  id = test$id,
  prob = y.pred
)

names(submission)[names(submission) == "prob.s0"] <- "prob"

write_csv(submission, 'mysubmission.csv')

