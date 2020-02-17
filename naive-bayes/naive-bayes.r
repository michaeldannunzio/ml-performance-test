###################################################################
# Project 4
# Part 2 -- Naive Bayes
# CS 4375.501
# Zain Husain & Michael D'Annunzio
###################################################################

# Load titanic csv into R
titanic_project <- read.csv("../titanic_project.csv")

# make copy and change columns to factors
tp <- titanic_project
tp$survived <- as.factor(tp$survived)
tp$sex <- as.factor(tp$sex)
tp$pclass <- as.factor(tp$pclass)

# create train and test sets
train <- tp[1:900,]
test <- tp[901:1046,]

# load e1071 library for Naive Bayes
# install.packages("e1071")
library(e1071)

# variables for calculating metrics
tp <- 0
tn <- 0
fp <- 0
fn <- 0

# start timing
start <- proc.time()

# Create Naive Bayes model from train data and view model
nb1 <- naiveBayes(survived ~ pclass+age+sex , data = train)

# predict on test data
predictions_class <- predict(nb1, newdata=test, type="class")

# predictions as integer vector
predVec <- as.integer(predictions_class)

# find tp, tn, fp, fn
for (x in 1:146)
{
  if (predVec[x] == 1) {if (test$survived[x] == 0) tn <- tn + 1 else fn <- fn + 1}
  if (predVec[x] == 2) {if (test$survived[x] == 1) tp <- tp + 1 else fp <- fp + 1}
}

# Results of confusion matrix
print(paste("TP: ", tp))
print(paste("TN: ", tn))
print(paste("FP: ", fp))
print(paste("FN: ", fn))

# calculate and print metrics
ac <- ((tp + tn) / nrow(test))
print(paste("Accuracy: ", ac))
se <- (tp / (tp + fn))
print(paste("Sensitivity: ", se))
sp <- (tn / (tn + fp))
print(paste("Specificity: ", sp))

# stop timing and print time elapsed
print(proc.time() - start)

# view model, confusion matrix, and statistics
nb1
confusionMatrix(predictions_class, test$survived, positive = "1")
