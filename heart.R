#------------------------------------------------ Data Loading -------------------------------------------------

library(dplyr)
library(DataExplorer)
library(ggplot2)
library(e1071)
library(kernlab)
library(lattice)
library(caret)
library(rpart)
library(rpart.plot)
library(party)
library(aod)
library(pscl)
library(caTools)

#Import Dataset
dataset = read.csv('heart.csv')
dataset <- read.csv("Heart disease/heart.csv", na.strings = "?", sep=",")
dataset

# View dataset
dim(dataset) # dimension
str(dataset) # structure
head(dataset) # first 6 records

plot_str(dataset)
plot_density(dataset)
#--------------------------------------------- Data Preprocessing ----------------------------------------------
# Checking for missing values
sum(is.na(dataset))
colSums(is.na(dataset))
plot_missing(dataset)

# Taking care of missing data

dataset$age = ifelse(is.na(dataset$age),
                     ave(dataset$age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$age)

dataset$trestbps = ifelse(is.na(dataset$trestbps),
                          ave(dataset$trestbps, FUN = function(x) mean(x, na.rm = TRUE)),
                          dataset$trestbps)

dataset$chol = ifelse(is.na(dataset$chol),
                      ave(dataset$chol, FUN = function(x) mean(x, na.rm = TRUE)),
                      dataset$chol)

# Detecting Outliers

boxplot(dataset$age)
boxplot(dataset$trestbps)
boxplot(dataset$chol)
boxplot(dataset$thalach)
boxplot(dataset$oldpeak)

plot_histogram (dataset$trestbps)
plot_histogram (dataset$chol)
plot_histogram (dataset$thalach)
plot_histogram (dataset$oldpeak)

# Treatement of Outliers 

T_log = log(dataset)
dataset$trestbps = T_log$trestbps
dataset$chol = T_log$chol

plot_histogram(dataset$trestbps)
plot_histogram(dataset$chol)

#correlation between variables
plot_correlation(dataset,'continuous')
# Checking Class distribution
prop.table(table(dataset$target))


dataset$target <- factor(dataset$target, levels=c(0,1), labels=c("1", "2"))
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
set.seed(123)
split = sample.split(dataset$target, SplitRatio = 0.7)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
prop.table(table(training_set$target))
prop.table(table(test_set$target))
#------------------------------------------------ Naive Bayes --------------------------------------------------
# Building classifier Naive Bayes

Naive_Classifier = naiveBayes(x = training_set[ ,-14], y = training_set$target )

# Using the classifier on training dataset to test the predictions
Naive_pred = predict(Naive_Classifier, newdata = training_set[ ,-14])
cm1 = table(training_set$target, Naive_pred)
cm1
summary(Naive_Classifier)
Accuracy = sum(diag(cm1))/sum(cm1)
Accuracy

# Using the classifier on test dataset 

Naive_pred1 = predict(Naive_Classifier1, newdata = test_set[ ,-14])
cm2 = table(test_set$target, Naive_pred1)
cm2
Accuracy = sum(diag(cm2))/sum(cm2)
Accuracy

# Using the classifier on training data set test the predictions
Naive_pred_raw = predict (Naive_Classifier, newdata = training_set[ ,-14], type = "raw" )
Naive_pred_class = predict (Naive_Classifier, newdata = training_set[ ,-14], type = "class" )
Naive_pred_class

s = cbind(Naive_pred_raw, Naive_pred_class)
head(s)

# Variation 2 - To apply Laplacian Smoothing

Naive_Classifier2 = naiveBayes(x = training_set[ ,-14], y = training_set$target, laplace=1 )

Naive_pred2 = predict(Naive_Classifier2, newdata = training_set[ ,-14])
cm3 = table(training_set$target, Naive_pred2)
cm3
Accuracy = sum(diag(cm3))/sum(cm3)
Accuracy

Naive_pred2 = predict(Naive_Classifier2, newdata = test_set[ ,-14])
cm4 = table(training_set$target, Naive_pred2)
cm4
Accuracy = sum(diag(cm4))/sum(cm4)
Accuracy


Naive_pred_raw1 = predict (Naive_Classifier2, newdata = training_set[ ,-14], type = "raw", threshold = 0.001, eps = 0 )
Naive_pred_class1 = predict (Naive_Classifier2, newdata = training_set[ ,-14], type = "class" )
Naive_pred_class1

s1 = cbind(Naive_pred_raw1, Naive_pred_class1)
head(s)

#training Naive Bayes model using {caret} package with 10 fold cross validation
NBclassifierCaretCV = train(x= training_set[,-14],y=training_set$target, 'nb', trControl = trainControl(method ='cv', number = 10))
CVtrainDataset = predict (NBclassifierCaretCV, newdata = training_set[,-14])
# Confusion matrix and a summary / using caret package
confusionMatrix(data = CVtrainDataset, training_set$target)


#testing Naive Bayes model using {caret} package with 10 fold cross validation
NBclassifierCaretCV = train(x= test_set[,-14],y=test_set$target, 'nb', trControl = trainControl(method ='cv', number = 10))
CVtrainDataset = predict (NBclassifierCaretCV, newdata = test_set[,-14])
# Confusion matrix and a summary / using caret package
confusionMatrix(data = CVtrainDataset, test_set$target)

#----------------------------------------------- Decision Tree -------------------------------------------------

# install.packages('rpart') --> (Recursive Partitioning And Regression Trees) and the R implementation of the CART algorithm
# install.packages("rpart.plot")
# install.packages("ROCR")
# install.packages("gplots")



#----------- Split with Gini Index -------------------------

tree = rpart(target~ ., data=training_set)
print(tree)
prp(tree) # plot Rpart Model
prp (tree, type = 5, extra = 2)
rpart.plot(tree, extra = 104, nn = TRUE)

#for Training accuracy
Predict = predict(tree, training_set, type = "class")
Predict

Confusion_matrix = table(Predict, training_set$target)
print(Confusion_matrix)

accuracy = sum(diag(Confusion_matrix))/sum(Confusion_matrix)*100
accuracy

#for Testing accuracy
Predict = predict(tree, test_set, type = "class")
Predict

Confusion_matrix = table(Predict, test_set$target)
print(Confusion_matrix)

accuracy = sum(diag(Confusion_matrix))/sum(Confusion_matrix)*100
accuracy

#----------- Split with entropy information -----------------

ent_Tree = rpart(target~ ., data=training_set, method="class", parms=list(split="information"))
prp(ent_Tree)
plotcp(ent_Tree)

#for Training accuracy
Predict = predict(ent_Tree, training_set, type = "class")
Predict

Confusion_matrix = table(Predict, training_set$target)
print(Confusion_matrix)

accuracy = sum(diag(Confusion_matrix))/sum(Confusion_matrix)*100
accuracy

#for Testing accuracy
Predict = predict(ent_Tree, test_set, type = "class")
Predict

Confusion_matrix = table(Predict, test_set$target)
print(Confusion_matrix)

accuracy = sum(diag(Confusion_matrix))/sum(Confusion_matrix)*100
accuracy
#-------------- Split with parameter settings ------------------

tree_with_params = rpart(target~ ., data=training_set, method="class", minsplit = 1, minbucket = 10, cp = -1)
prp (tree_with_params)
print(tree_with_params)
summary(tree_with_params)
plot(tree_with_params)
text(tree_with_params)
plotcp(tree_with_params)

#for Training accuracy
Predict = predict(tree_with_params, training_set, type = "class")
Predict

Confusion_matrix = table(Predict, training_set$target)
print(Confusion_matrix)

accuracy = sum(diag(Confusion_matrix))/sum(Confusion_matrix)*100
accuracy

#for Testing accuracy
Predict = predict(tree_with_params, test_set, type = "class")
Predict

Confusion_matrix = table(Predict, test_set$target)
print(Confusion_matrix)

accuracy = sum(diag(Confusion_matrix))/sum(Confusion_matrix)*100
accuracy

controlParameters = trainControl(method ='cv', number = 10)
gbmmodel = train(target~., data = training_set, method = "gbm", trControl = controlParameters)
CVtrainDataset = predict (gbmmodel, newdata = training_set[,-14])
confusionMatrix(data = CVtrainDataset, training_set$target)

controlParameters = trainControl(method ='cv', number = 10)
gbmmodel = train(target~., data = test_set, method = "gbm", trControl = controlParameters)
CVtrainDataset = predict (gbmmodel, newdata = test_set[,-14])
confusionMatrix(data = CVtrainDataset, test_set$target)

#------------------------------------------ Support Vector Machine ----------------------------------------------

# ~~~~~~~~~~~~~~  Default SVM Model using the RBF kernel ~~~~~~~~~~~~~~~~~

svm_rbf <- svm(target~., data = training_set)
summary(svm_rbf)
svm_rbf$gamma

# Confusion Matrix
pred = predict (svm_rbf, training_set)
pred

cm = table(Predicted = pred, Actual = training_set$target)
cm
accuracy = sum(diag(cm))/sum(cm)*100
accuracy


pred = predict (svm_rbf, test_set)
pred
cm = table(Predicted = pred, Actual = test_set$target)
cm
accuracy = sum(diag(cm))/sum(cm)*100
accuracy


confusionMatrix(table(pred, test_set$target))

# ~~~~~~~~~~~~~~~~~~~~   SVM model using the Linear model  ~~~~~~~~~~~~~~~~~~~~~
svm_linear = svm (target~., data = training_set, kernel = "linear")
summary (svm_linear)

# Confusion Matrix
pred = predict (svm_linear, training_set)
pred
cm = table(Predicted = pred, Actual = training_set$target)
cm
accuracy = sum(diag(cm))/sum(cm)*100
accuracy

pred = predict (svm_linear, test_set)
pred
cm = table(Predicted = pred, Actual = test_set$target)
cm
1-sum(diag(cm))/sum(cm)
accuracy = sum(diag(cm))/sum(cm)*100
accuracy
# ~~~~~~~~~~~~~~~~~~~~   SVM model using sigmoid kernal  ~~~~~~~~~~~~~~~~~~~~~
svm_sigmoid = svm (target~., data = training_set, kernel = "sigmoid")
summary (svm_sigmoid)

# Confusion Matrix
pred = predict (svm_sigmoid, training_set)
cm = table(Predicted = pred, Actual = training_set$target)
cm
accuracy = sum(diag(cm))/sum(cm)*100
accuracy

pred = predict (svm_sigmoid, test_set)
cm = table(Predicted = pred, Actual = test_set$target)
cm
accuracy = sum(diag(cm))/sum(cm)*100
accuracy


# ~~~~~~~~~~~~~~~~~~~~   SVM model using polynomial kernal  ~~~~~~~~~~~~~~~~~~~~~
svm_polynomial = svm (target~., data = training_set, kernel = "poly")
summary (svm_polynomial)

# Confusion Matrix
pred = predict (svm_polynomial, test_set)
cm_poly = table(Predicted = pred, Actual = test_set$target)
cm_poly

accuracy = sum(diag(cm_poly))/sum(cm_poly)*100
accuracy

pred = predict (svm_polynomial, training_set)
cm_poly = table(Predicted = pred, Actual = training_set$target)
cm_poly

accuracy = sum(diag(cm_poly))/sum(cm_poly)*100
accuracy


# ~~~~~~~~~~~~~~~~~~~~  Model Tuning  ~~~~~~~~~~~~~~~~~~~~
set.seed(123)
# tune function tunes the hyperparameters of the model using grid search method
tuned_model = tune(svm, target~., data=training_set,
                   ranges = list(epsilon = seq (0, 1, 0.1), cost = 2^(0:2)))
plot (tuned_model)
summary (tuned_model)

opt_model = tuned_model$best.model
summary(opt_model)

# Building the best model
svm_best <- svm (target~., data = training_set, epsilon = 0, cost = 4)
summary(svm_best)

pred = predict (svm_best, training_set)
cm_best = table(Predicted = pred, Actual = training_set$target)
cm_best

accuracy = sum(diag(cm_best))/sum(cm_best)*100
accuracy

pred = predict (svm_best, test_set)
cm_best = table(Predicted = pred, Actual = test_set$target)
cm_best

accuracy = sum(diag(cm_best))/sum(cm_best)*100
accuracy

#-------Cross Validation-----------
controlParameters = trainControl(method ='cv', number = 10)
modelSvmCV = train(target~., data = training_set, method = "svmRadial", trControl = controlParameters)
CVtrainDataset = predict (modelSvmCV, newdata = training_set[,-14])
confusionMatrix(data = CVtrainDataset, training_set$target)

controlParameters = trainControl(method ='cv', number = 10)
modelSvmCV = train(target~., data = test_set, method = "svmRadial", trControl = controlParameters)
CVtrainDataset = predict (modelSvmCV, newdata = test_set[,-14])
confusionMatrix(data = CVtrainDataset, test_set$target)
#-------------------------------------------- Logistic Regression -----------------------------------------------
# Building classifier Logistic Regression Training_set
logit_Regression = glm(target ~.,
                       training_set,
                       family = binomial)
summary(logit_Regression)

coef(logit_Regression)
confint(logit_Regression)
anova(logit_Regression, test='Chisq')

# Mcfadden R2


pR2(logit_Regression)

#test the model significance

wald.test(b=coef(logit_Regression),Sigma = vcov(logit_Regression),Terms = 4:6)

# Predicting the Test set results
logistic_pred = predict(logit_Regression, type = 'response', training_set[ ,-14] )
y_pred = ifelse(logistic_pred > 0.5, 1, 0)
cm = table(training_set$target, y_pred)#confusion Matrix
cm
Accuracy = sum(diag(cm))/sum(cm)
Accuracy

logistic_pred1 = predict(logit_Regression, type = 'response', test_set[ ,-14] )
y_pred1 = ifelse(logistic_pred1 > 0.5, 1, 0)
cm1 = table(test_set$target, y_pred1)#confusion Matrix
cm1
Accuracy = sum(diag(cm1))/sum(cm1)
Accuracy

controlParameters = trainControl(method ='cv', number = 10)
glmmodel = train(target~., data = training_set, method = "glm", trControl = controlParameters)
CVtrainDataset = predict (gbmmodel, newdata = training_set[,-14])
confusionMatrix(data = CVtrainDataset, training_set$target)

controlParameters = trainControl(method ='cv', number = 10)
glmmodel = train(target~., data = test_set, method = "glm", trControl = controlParameters)
CVtrainDataset = predict (gbmmodel, newdata = test_set[,-14])
confusionMatrix(data = CVtrainDataset, test_set$target)
