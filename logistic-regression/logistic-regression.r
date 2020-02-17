###################################################################
# Project 4
# Part 1 -- Logistic Regression
# CS 4375.501
# Zain Husain & Michael D'Annunzio
###################################################################

###################################################################
# Part 1: Using R function to create logistic regression model
###################################################################
# Add necessary libraries and datasets
library(HSAUR)
data("plasma")

# attch plasma for easy use
attach(plasma)

# create Logistic regression using pre-made R functions
glm1 <- glm(ESR ~ fibrinogen, family = binomial, data =plasma)

# print coefficients
print("Premade R Logistic Regression:")
print(coefficients(glm1))

# Data exploration
summary(glm1)
str(plasma)
table(ESR)
table(fibrinogen)

# Data visualization
hist(fibrinogen)
cdplot(fibrinogen, ESR)

###################################################################
# Part 2: Hand inplemented Logistic regression
###################################################################

# define sigmoid function
sigmoid <- function(z){
  1.0 / (1+exp(-z))
}

# set up weight vector, label vector, and data matrix
weights <- c(1, 1)
data_matrix <- cbind(rep(1, nrow(plasma)), fibrinogen)
labels <- as.integer(ESR) - 1

# Gradient Descent from Scratch
learning_rate <- 0.001

# start timing
start <- proc.time()

# learning loop
for (i in 1:500000){
  prob_vector <- sigmoid(data_matrix %*% weights)
  error <- labels - prob_vector
  weights <- weights + learning_rate * t(data_matrix) %*% error
}

# finish timing and print time elapsed
print(proc.time() - start)

# print coefficients
print(weights)
