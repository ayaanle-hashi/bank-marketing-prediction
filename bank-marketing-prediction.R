library(dplyr)
library(tidyverse)
library(caret)
library(ranger)
library(xgboost)
library(ROSE)

options(scipen = 999)

df = read.csv("bank-full.csv",sep = ";")
 
# Viewing if there are any null values within dataset
sapply(df, function(x) sum(is.na(x))) 


# Cleaning dataset

df$default = ifelse(df$default == 'yes',1,0)
df$housing = ifelse(df$housing == 'yes',1,0)
df$loan = ifelse(df$loan == 'yes',1,0)
df$y = ifelse(df$y == 'yes',1,0)
df$month = str_to_sentence(df$month)
df$month = match(df$month,month.abb)

# Remove outliers from continous variables 

remove_outlier = function(dataframe,column){
  average = mean(dataframe[[column]])
  std = sd(dataframe[[column]])
  cutoff = 3 * std
  lower = average - cutoff
  upper = average + cutoff
  new = dataframe[dataframe[[column]] > lower & dataframe[[column]] < upper,]
  return(new)
}

df = remove_outlier(df,'balance')
df = remove_outlier(df,'duration')


# Lets view if job and month has any impact on subscription rates

subscription_rate_month = df %>%
  group_by(month) %>%
  summarise(rate =mean(y))


ggplot(subscription_rate_month, aes(x=factor(month),y=rate)) +
  geom_bar(stat = "identity",width=0.7, fill="steelblue") +
  xlab("Month") +
  ggtitle("Month vs Subscription Rates") +
  theme(plot.title = element_text(hjust = 0.5))


subscription_rate_job = df %>%
  group_by(job) %>%
  summarise(rate =mean(y))


ggplot(subscription_rate_job, aes(x=factor(job),y=rate)) +
  geom_bar(stat = "identity",width=0.7, fill="steelblue") +
  theme(axis.text.x=element_text(angle=90,hjust=1)) +
  xlab("Job") +
  ggtitle("Job vs Subscription Rates") +
  theme(plot.title = element_text(hjust = 0.5))

# No visible difference between the different months and subscription rates.
# However there is a difference between jobs and subscription rates with Retired people and students 
# Having a subscription rate of 24% and 28% respectively.
# I will drop the month variable and create a binary feature for the jobs highlighting retired and students.

df = subset(df, select = -c(month,pdays))
df$job = ifelse(df$job == "student" | df$job == "retired",1,0)

dummy_df = dummyVars("~ .",data = df)
df = predict(dummy_df,newdata = df)
df = data.frame(df)

train_index = createDataPartition(y=df$y,p=0.7,list = FALSE)
df_train = df[train_index,]
df_test = df[-train_index,]

model = glm(y~.,family = binomial(link = "logit"),data=df_train)
summary(model)

data = df_test[ ,!(colnames(df_test) == "y")]
target = df_test[,"y"]

probabilities = model %>% predict(data,type="response")
predictions = as.factor(ifelse(probabilities >0.5,1,0))

precision <- posPredValue(predictions, as.factor(target), positive="1")
recall <- sensitivity(predictions, as.factor(target), positive="1")

# Precision of 0.63 and Recall of 0.33

# Lets try sampling to see if this can be improved.

comb_sample = ovun.sample(y~.,data=df_train,method="both")$data

model = glm(y~.,family = binomial(link = "logit"),data=comb_sample)

probabilities = model %>% predict(data,type="response")
predictions = as.factor(ifelse(probabilities >0.5,1,0))
precision <- posPredValue(predictions, as.factor(target), positive="1")
recall <- sensitivity(predictions, as.factor(target), positive="1")

# Precision of 0.39 and Recall of 0.78.
# Model is simply classifying most instances as the positive class leading to an increase in false positives.

# Lets try using more complicated algorithms.

rf_model = ranger(y~.,data=df_train,classification = TRUE)
rf_pred = predict(rf_model, data = df_test)
predictions = as.factor(rf_pred$predictions)
precision = posPredValue(predictions, as.factor(target), positive="1")
recall = sensitivity(predictions, as.factor(target), positive="1")

# Precision of 0.64 and recall of 0.33

xgb = xgboost(data=data.matrix(df_train[,!(colnames(df_train) == "y")]),
        label=df_train$y,
        objective = "binary:logistic",
        scale_pos_weight = 7,
        max.depth = 20,
        eta = 1,
        nrounds = 400)
preds = predict(xgb, data.matrix(data))
pred_classes = as.factor(ifelse(preds >0.5,1,0))

precision <- posPredValue(pred_classes, as.factor(target), positive="1")
recall <- sensitivity(pred_classes, as.factor(target), positive="1")

# Precision of 0.49 Recall of 0.46

# Random Forrest had a higher precision but lower recall than xgboost.
# Model recommendation would depend on usecase of model and wether lower false positives or lower false negatives is prefered. 
