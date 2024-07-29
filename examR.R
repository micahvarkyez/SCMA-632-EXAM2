# Load necessary packages and libraries
install.packages("tidyverse")
install.packages("caret")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("pROC")
install.packages("e1071")
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)
library(e1071)
setwd("C:\\Users\\Dell\\Desktop\\MICAH")
data <- read.csv("bank-additional-full.csv", sep = ";")
# Inspect the dataset
str(data)
summary(data)

# Check the first few rows of the dataset
head(data)

# Check for missing values
sum(is.na(data))

# Convert categorical variables to factors
data <- data %>% mutate_if(is.character, as.factor)

# Inspect the data structure
str(data)

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
train_index <- createDataPartition(data$y, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Logistic Regression Model
logistic_model <- glm(y ~ ., data = train_data, family = binomial)

# Predictions and probabilities
logistic_preds <- predict(logistic_model, test_data, type = "response")
logistic_pred_class <- ifelse(logistic_preds > 0.5, "yes", "no")

# Confusion matrix for Logistic Regression
logistic_conf_matrix <- confusionMatrix(data = as.factor(logistic_pred_class), reference = test_data$y)

# Calculate AUC-ROC for Logistic Regression
roc_logistic <- roc(test_data$y, logistic_preds)
auc_logistic <- auc(roc_logistic)
plot(roc_logistic)

# Decision Tree Model
tree_model <- rpart(y ~ ., data = train_data, method = "class")
# Predictions
tree_preds <- predict(tree_model, test_data, type = "class")

# Confusion matrix for Decision Tree
tree_conf_matrix <- confusionMatrix(data = tree_preds, reference = test_data$y)

# Calculate AUC-ROC for Decision Tree
tree_probs <- predict(tree_model, test_data, type = "prob")[,2]
roc_tree <- roc(test_data$y, tree_probs)
auc_tree <- auc(roc_tree)
plot(roc_tree)

# Visualize the Decision Tree
rpart.plot(tree_model)

# Metrics for Logistic Regression
logistic_accuracy <- logistic_conf_matrix$overall['Accuracy']
logistic_precision <- posPredValue(as.factor(logistic_pred_class), test_data$y, positive = "yes")
logistic_recall <- sensitivity(as.factor(logistic_pred_class), test_data$y, positive = "yes")
logistic_f1 <- 2 * ((logistic_precision * logistic_recall) / (logistic_precision + logistic_recall))

# Metrics for Decision Tree
tree_accuracy <- tree_conf_matrix$overall['Accuracy']
tree_precision <- posPredValue(tree_preds, test_data$y, positive = "yes")
tree_recall <- sensitivity(tree_preds, test_data$y, positive = "yes")
tree_f1 <- 2 * ((tree_precision * tree_recall) / (tree_precision + tree_recall))

# Output metrics
metrics <- data.frame(
  Model = c("Logistic Regression", "Decision Tree"),
  Accuracy = c(logistic_accuracy, tree_accuracy),
  Precision = c(logistic_precision, tree_precision),
  Recall = c(logistic_recall, tree_recall),
  F1_Score = c(logistic_f1, tree_f1),
  AUC_ROC = c(auc_logistic, auc_tree)
)

print(metrics)

# Provide interpretations of the results
summary(logistic_model)
rpart.plot(tree_model)