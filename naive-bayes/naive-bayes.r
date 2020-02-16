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
# install.packages("e1071")
# load e1071 library for Naive Bayes
library(e1071)
# Create Naive Bayes model from train data and view model
nb1 <- naiveBayes(survived ~ pclass+age+sex , data = train)
nb1
# predict on test data
predictions_class <- predict(nb1, newdata=test, type="class")
# predictions_raw <- predict(nb1, newdata=test, type="raw")
# head(predictions_raw)
# view confusion matrix and statistics
confusionMatrix(predictions_class, test$survived, positive = "1")
