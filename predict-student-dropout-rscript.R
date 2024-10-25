# SOURCE CODE 

# Section 1: Setting Up the Environment

# Subsection 1.1: Installing and Loading the Necessary Packages 

# Install packages
install.packages("dplyr", dependencies = TRUE) 
install.packages("missForest", dependencies = TRUE)
install.packages("DataExplorer", dependencies = TRUE)
install.packages("caret", dependencies = TRUE)
install.packages("ROCR", dependencies = TRUE)
install.packages("ggplot2", dependencies = TRUE)
install.packages("caTools", dependencies = TRUE)
install.packages("xgboost", dependencies = TRUE)
install.packages("randomForest", dependencies = TRUE)
install.packages("pROC", dependencies = TRUE)
install.packages("MLmetrics", dependencies = TRUE)
install.packages("rpart", dependencies = TRUE)
install.packages("reshape2")

# Load the installed packages 
library(dplyr)
library(missForest)
library(DataExplorer)
library(caret)
library(ROCR)
library(ggplot2)
library(caTools)
library(xgboost)
library(randomForest)
library(pROC)
library(MLmetrics)
library(rpart)
library(reshape2)

# Subsection 1.2: Importing the dataset 
ds <- read.csv("predict-student-dropout.csv")


# Section 2.0: Data Preprocessing 

# Subsection 2.1 Initial exploration of the dataset 

# Show initial entries of the dataset for a quick preview
head(ds)

# Examine the dataset's structure, including data types and column information
str(ds)

# Summarize the dataset with statistics for each column
summary(ds)

# Check for the dimensions of the dataset, namely the count of rows and columns
dim(ds)

# Graphically represent the presence of missing entries across all columns
plot_missing(ds)

# Determine the count of distinct entries in each column
sapply(ds, function(x) length(unique(x)))

# Subsection 2.2: Data Cleaning 

# Subsection 2.2.1: Renaming of the Column Label 

# Change the column label from 'Nacionality' to the correct spelling 'Nationality'
ds <- ds %>% 
  rename(Nationality = Nacionality)

# Subsection 2.2.2: Conversion of Variables Into Factor Data 

# Specify columns to be converted into factor data type 
categorical_columns <- c(
  "Marital.status", "Application.mode", "Application.order", "Course", "Daytime_evening.attendance",
  "Previous.qualification", "Nationality", "Mother.qualification", "Father.qualification",
  "Mother.occupation", "Father.occupation", "Displaced", "Educational.special.needs", "Debtor",
  "Tuition.fees.up.to.date", "Gender", "Scholarship.holder", "Age.at.enrollment", "International",
  "Curricular.units.1st.sem..credited.", "Curricular.units.1st.sem..enrolled.",
  "Curricular.units.1st.sem..evaluations.", "Curricular.units.1st.sem..approved.",
  "Curricular.units.1st.sem..without.evaluations.", "Curricular.units.2nd.sem..credited.",
  "Curricular.units.2nd.sem..enrolled.", "Curricular.units.2nd.sem..evaluations.",
  "Curricular.units.2nd.sem..approved.", "Curricular.units.2nd.sem..without.evaluations.", "Target"
)

# Categorize selected columns as factors for analysis
ds[categorical_columns] <- lapply(ds[categorical_columns], factor)

# Subsection 2.2.3: Detection and Treatment of Outliers

# Identify continuous variables to examine for potential outliers
continuous_variables <- names(ds)[sapply(ds, is.numeric)]
print(continuous_variables)

# Create a function to pinpoint outliers based on the interquartile range method
outliers_function <- function(data, column) {
  Q1 <- quantile(data[[column]], 0.25)
  Q3 <- quantile(data[[column]], 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 3 * IQR
  upper_bound <- Q3 + 3 * IQR
  outliers <- data[data[[column]] < lower_bound | data[[column]] > upper_bound, ]
  return(nrow(outliers))
}

# Count and report the number of outliers in each continuous column
outliers_count <- sapply(continuous_variables, outliers_function, data = ds)
print(outliers_count)

# Implement a procedure to adjust outlier values to the column mean, thereby mitigating their effect
columns_to_treat <- c('Previous.qualification..grade.', 'Admission.grade')
mean_treatment_function <- function(data, column) {
  Q1 <- quantile(data[[column]], 0.25)
  Q3 <- quantile(data[[column]], 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 3 * IQR
  upper_bound <- Q3 + 3 * IQR
  
  is_outlier <- data[[column]] < lower_bound | data[[column]] > upper_bound
  mean_value <- mean(data[[column]], na.rm = TRUE)
  data[is_outlier, column] <- mean_value
  
  return(data)
}

# Apply the outlier adjustment function to the relevant columns.
for (column in columns_to_treat) {
  ds <- mean_treatment_function(ds, column)
}

# Reevaluate the columns to confirm the adjustment of outliers
post_outliers <- sapply(columns_to_treat, outliers_function, data = ds)
print(post_outliers)

# Subsection 2.2.5: Treatment of Problematic Entries

# Apply filters to identify and label specific academic data as missing (NA).
problematic_columns <- c(
  "Curricular.units.1st.sem..credited.", "Curricular.units.1st.sem..enrolled.", 
  "Curricular.units.1st.sem..evaluations.", "Curricular.units.1st.sem..approved.", 
  "Curricular.units.1st.sem..without.evaluations.", 
  "Curricular.units.2nd.sem..credited.", "Curricular.units.2nd.sem..enrolled.", 
  "Curricular.units.2nd.sem..evaluations.", "Curricular.units.2nd.sem..approved.", 
  "Curricular.units.2nd.sem..without.evaluations."
)

# Process the dataset by marking academic records as missing (NA) when all specified academic  
# columns have zero values, specifically for rows where the target status is 'Graduate' or 'Enrolled'. 
# Then, remove the temporary column used for calculations.
ds <- ds %>%
  mutate(academic_sum = rowSums(select(., all_of(problematic_columns)) == 0)) %>%
  mutate_at(vars(all_of(problematic_columns)), ~ifelse(Target %in% c('Graduate', 'Enrolled') 
                                                       & academic_sum == length(problematic_columns), NA, .)) %>%
  select(-academic_sum)  # Remove the temporary helper column

# Preview the modified dataset where certain records are now marked as missing.
filtered_indexes <- which(ds$Target %in% c('Graduate', 'Enrolled') & 
                            rowSums(ds[problematic_columns], na.rm = TRUE) == 0)
print(head(ds[filtered_indexes, problematic_columns]))

# Visualize missing data across all columns of the dataset.
plot_missing(ds)

# Implement missForest for imputing the missing values identified
seed <- 123
set.seed(seed)
ds_cleaned <- missForest(ds, maxiter = 10, ntree = 100, variablewise = TRUE, decreasing = TRUE)
ds_cleaned <- missForest(ds)$ximp

# Save the dataset with imputed values as a new file
write.csv(ds_cleaned, file = "cleaned_data.csv", row.names = FALSE)

# Reload the dataset with the imputed values
cleaned_ds <- read.csv("cleaned_data.csv")

# Section 3.0: Exploratory Data Analysis (EDA) 

# Subsection 3.1: Examining the Target Column

# Count and display the frequency of each target category
table(cleaned_ds$Target)

# Visual representation of target category distribution using a bar chart with eye-friendly colors
ggplot(cleaned_ds, aes(x = Target, fill = Target)) + 
  geom_bar(stat = "count") + 
  labs(title = "Distribution of Each Target Categories") +
  scale_fill_manual(values = c("#8EC8E6", "#FDB0C0", "#B3DE69")) + # Soft blue, soft pink, soft green
  theme_minimal()

# Represent target category proportions using a pie chart.
ggplot(cleaned_ds, aes(x = "", fill = Target)) + 
  geom_bar(width = 1, stat = "count") + 
  coord_polar("y", start = 0) + 
  labs(title = "Target Category Proportions") + 
  scale_fill_manual(values = c("#8EC8E6", "#FDB0C0", "#B3DE69")) +
  geom_text(aes(label = scales::percent(..count.. / sum(..count..))), 
            stat = "count", position = position_stack(vjust = 0.5)) +
  theme_void() + 
  theme(legend.position = "bottom")

# Subsection 3.2: Exploring Correlations in Data

# Generate a matrix to visualize the correlations between different variables in the dataset
plot_correlation(cleaned_ds)

# Section 4.0: Implementing the Model

# Subsection 4.1: Preparing the Data for Modelling

# Subsection 4.1.1: Preparing the Target Variable

# Show distinct categories in the target column
unique(cleaned_ds$Target)

# Recode target column values to numerical labels: Graduate as 0, Enrolled as 1, Dropout as 2
cleaned_ds$Target <- factor(cleaned_ds$Target, levels = c("Graduate", "Enrolled", "Dropout"), 
                            labels = c(0, 1, 2))

# Convert categorical columns to factors for modeling
columns <- c(
  "Marital.status", "Application.mode", "Application.order", "Course", "Daytime_evening.attendance",
  "Previous.qualification", "Nationality", "Mother.qualification", "Father.qualification",
  "Mother.occupation", "Father.occupation", "Displaced", "Educational.special.needs", "Debtor",
  "Tuition.fees.up.to.date", "Gender", "Scholarship.holder", "Age.at.enrollment", "International",
  "Curricular.units.1st.sem..credited.", "Curricular.units.1st.sem..enrolled.",
  "Curricular.units.1st.sem..evaluations.", "Curricular.units.1st.sem..approved.",
  "Curricular.units.1st.sem..without.evaluations.", "Curricular.units.2nd.sem..credited.",
  "Curricular.units.2nd.sem..enrolled.", "Curricular.units.2nd.sem..evaluations.",
  "Curricular.units.2nd.sem..approved.", "Curricular.units.2nd.sem..without.evaluations.", "Target"
)

cleaned_ds[columns] <- lapply(cleaned_ds[columns], factor)

# Check the recoded target column values
unique(cleaned_ds$Target)

# Subsection 4.1.2: Standardizing Numerical Data

# Identify and list columns with numeric data
numeric_columns <- names(cleaned_ds)[sapply(cleaned_ds, is.numeric)]

# Standardize these columns for uniform scaling
cleaned_ds[numeric_columns] <- lapply(cleaned_ds[numeric_columns], scale)

# Overview of standardized data
summary(cleaned_ds[numeric_columns])

# Subsection 4.1.3: Splitting Data into Training and Testing Sets

# Divide the dataset into training and testing sets using a 70:30 ratio
set.seed(321)
split = sample.split(cleaned_ds$Target, SplitRatio = 0.7)
training_data = subset(cleaned_ds, split == TRUE)
testing_data = subset(cleaned_ds, split == FALSE)

# Subsection 4.2: Building an XGBoost Model 

# Subsection 4.2.1: Initial Model Training

# Setting default parameters for XGBoost
params <- list(
  booster = "gbtree",
  num_class = 3,
  objective = "multi:softprob",
  verbosity = 1
)

# Converting categorical variables in training data to numeric
training_data[] <- lapply(training_data, function(x) if (is.factor(x)) as.numeric(x) else x)

# Identifying the position of the target variable in the dataset
target_column_index <- which(colnames(training_data) == "Target")

# Preparing feature matrix and target vector for model input
features_matrix <- as.matrix(training_data[, -target_column_index, drop = FALSE])
labels_vector <- as.vector(training_data[, target_column_index]) - 1

# Creating DMatrix for XGBoost
xgb_data <- xgb.DMatrix(data = features_matrix, label = labels_vector)

# Executing the XGBoost training
set.seed(321)
xgb_base_model <- xgb.train(params = params, data = xgb_data, nrounds = 100)

# Assessing the Base Model on Training Data

# Predicting probabilities on training data
xgb_prob_predictions <- predict(xgb_base_model, as.matrix(training_data[, -target_column_index]))

# Transforming probabilities to predicted classes
xgb_class_predictions <- max.col(matrix(xgb_prob_predictions, nrow = 
                                          length(training_data$Target), byrow = TRUE))

# Preparing actual and predicted values for confusion matrix
actuals <- as.factor(training_data$Target)
predictions <- as.factor(xgb_class_predictions)

# Generating and displaying training data confusion matrix
confusionMatrix(predictions, actuals, mode = "everything")

# Model Evaluation on Test Data

# Converting test data factors to numeric
testing_data[] <- lapply(testing_data, function(x) if (is.factor(x)) as.numeric(x) else x)

# Making predictions on the test dataset
xgb_prob_predictions_test <- predict(xgb_base_model, as.matrix(testing_data[, -target_column_index]))

# Determining class labels from predicted probabilities
xgb_class_predictions_test <- max.col(matrix(xgb_prob_predictions_test, nrow = 
                                               length(testing_data$Target), byrow = TRUE)) - 1
xgb_class_predictions_test_adjusted <- xgb_class_predictions_test + 1

# Preparing actual and predicted values for test data confusion matrix
actuals_test <- as.factor(testing_data$Target)
predictions_test <- as.factor(xgb_class_predictions_test_adjusted)

# Generating and displaying test data confusion matrix
confusionMatrix(predictions_test, actuals_test, mode = "everything")

# Evaluating Model Using AUC Metric

# Organizing predicted probabilities into a matrix for AUC calculation
xgb_prob_matrix <- matrix(xgb_prob_predictions_test, nrow = length(testing_data$Target), byrow = TRUE)

# Labeling columns of the probability matrix for AUC calculation
colnames(xgb_prob_matrix) <- levels(as.factor(testing_data$Target))

# Calculating and displaying multi-class AUC
auc_result <- multiclass.roc(testing_data$Target, xgb_prob_matrix)
print(paste("Multi-class AUC:", auc_result$auc))

# Subsection 4.2.2: Hyperparameter Tuning

# Defining cross-validation and search strategy
train_control <- trainControl(method = "cv", number = 5, search = "random")

# Specifying the range of hyperparameters to explore
tune_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 5, 7, 9),
  eta = c(0.01, 0.05, 0.1, 0.2),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = c(0.6, 0.8, 1),
  min_child_weight = c(1, 2, 3, 4),
  subsample = c(0.7, 0.8, 0.9)
)

# Conducting hyperparameter tuning on the XGBoost model
xgb_model <- train(
  Target ~ ., 
  data = training_data, 
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = tune_grid, 
  nthread = 16
)

# Outputting the best hyperparameters found
print(xgb_model$bestTune)

# Training Data Evaluation of Tuned XGBoost Model

# Generating predictions on training data
xgb_predictions_rs <- predict(xgb_model, training_data)
xgb_predictions_rs <- round(xgb_predictions_rs)

# Matching prediction levels to actual target levels
xgb_predictions_rs <- as.factor(xgb_predictions_rs)
training_data$Target <- as.factor(training_data$Target)
levels(xgb_predictions_rs) <- levels(training_data$Target)

# Displaying confusion matrix for training data
confusionMatrix(xgb_predictions_rs, training_data$Target, mode = "everything")

# Test Data Evaluation of Tuned XGBoost Model

# Generating predictions on test data
xgb_predictions_rs_test <- predict(xgb_model, testing_data)
xgb_predictions_rs_test <- round(xgb_predictions_rs_test)

# Aligning prediction levels with actual target levels for test data
xgb_predictions_rs_test <- as.factor(xgb_predictions_rs_test)
testing_data$Target <- as.factor(testing_data$Target)
levels(xgb_predictions_rs_test) <- levels(testing_data$Target)

# Displaying confusion matrix for test data
xgb_metrics <- confusionMatrix(xgb_predictions_rs_test, testing_data$Target, mode = "everything")
xgb_metrics

# AUC Metric for Tuned XGBoost Model

# Predicting probabilities for test data and calculating AUC
xgb_prob_predictions_test <- predict(xgb_model, newdata = testing_data)
xgb_auc <- multiclass.roc(testing_data$Target, xgb_prob_predictions_test)
auc(xgb_auc)

# Subsection 4.2.3 Feature Importance Analysis

# Analyzing Feature Importance in XGBoost Model

# Retrieving the final XGBoost model
final_xgb_model <- xgb_model$finalModel

# Extracting and visualizing feature importance
feature_importance_matrix <- xgb.importance(model = final_xgb_model)
xgb.plot.importance(feature_importance_matrix)

# Subsection 4.3: Building a Random Forest Model

# Subsection 4.3.1: Initial Model Training

# Set up parameters for training without cross-validation
train_control <- trainControl(method = "none", verboseIter = T,  allowParallel = TRUE)

# Develop a Random Forest model using the training dataset
set.seed(321)
rf_model <- train(Target ~ ., data = training_data, method = "rf", trControl = train_control)

# Test the Random Forest model on training data and display the results using a confusion matrix
rf_predictions <- predict(rf_model, training_data)
confusionMatrix(rf_predictions, training_data$Target, mode = "everything")

# Assess the Random Forest model performance on test data and present the results in a confusion matrix
rf_predictions_test <- predict(rf_model, testing_data)
confusionMatrix(rf_predictions_test, testing_data$Target, mode = "everything")

# Evaluate model accuracy using Area Under Curve (AUC) metric
auc <- multiclass.roc(testing_data$Target, as.numeric(rf_predictions_test))
auc(auc)

# Subsection 4.3.2: Hyperparameter Tuning

# Set up random search for optimizing model hyperparameters with cross-validation
train_control <- trainControl(method = "cv", number = 2, search="random", allowParallel = TRUE)

# Execute random search to fine-tune the Random Forest model
set.seed(321)
rf_model_rs <- train(Target ~ ., data = training_data, method = "rf", trControl = train_control, 
                     metric = "Accuracy", tuneLength = 15)

# Print the details of the model
print(rf_model_rs)

# Test the tuned model on training data and present the results with a confusion matrix
rf_predictions_rs <- predict(rf_model_rs, training_data)
confusionMatrix(rf_predictions_rs, training_data$Target, mode = "everything")

# Assess the performance of the tuned model on test data using a confusion matrix
rf_predictions_rs_test <- predict(rf_model_rs, testing_data)
confusionMatrix(rf_predictions_rs_test, testing_data$Target, mode = "everything")

# Implement grid search method for further hyperparameter optimization
train_control <- trainControl(method = "cv", number = 10, search="grid", verboseIter = T,  allowParallel = TRUE)
grid <- expand.grid(mtry = seq(2, 36, by = 2))
set.seed(321)
rf_model_gs <- train(Target ~ ., data = training_data, method = "rf", trControl = train_control, 
                     metric = "Accuracy", tuneGrid = grid)

# Evaluate the grid-search optimized model on training data and display using a confusion matrix
rf_predictions_gs <- predict(rf_model_gs, training_data)
confusionMatrix(rf_predictions_gs, training_data$Target, mode = "everything")

# Test the model with test data and showcase results with a confusion matrix
rf_predictions_gs_test <- predict(rf_model_gs, testing_data)

# Present confusion matrix with detailed metrics
rf_metrics <- confusionMatrix(rf_predictions_gs_test, testing_data$Target, mode = "everything")
rf_metrics

# Measure model performance using Area Under the Curve
# Employ the multiclass.roc() function for AUC calculation
rf_auc <- multiclass.roc(testing_data$Target, as.numeric(rf_predictions_gs_test))
auc(rf_auc)

# Subsection 4.3.3: Feature Importance Analysis

# Assess and select key features based on their significance
# Determine the importance of each feature
feature_importance <- varImp(rf_model_gs, scale = FALSE)
print(feature_importance)

# Visualize the significance of each feature
plot(feature_importance)

# Subsection 4.4： Building a Decision Tree Model 

# Subsection 4.4.1: Initial Model Training

# Develop a Decision Tree model using the training dataset
set.seed(123)
baseline_decision_tree <- rpart(Target ~ ., data = training_data, method = "class")

# Training Data Evaluation of Baseline Model
baseline_predictions_train <- predict(baseline_decision_tree, training_data, type = "class")
confusionMatrix(baseline_predictions_train, training_data$Target, mode = "everything")

# Test Data Evaluation of Baseline Model
baseline_predictions_test <- predict(baseline_decision_tree, testing_data, type = "class")
confusionMatrix(baseline_predictions_test, testing_data$Target, mode = "everything")

# Subsection 4.4.2: Hyperparameter Tuning

# Set up random search for optimizing model hyperparameters with cross-validation
train_control_rs <- trainControl(method = "cv", number = 2, search = "random", allowParallel = TRUE)

# Execute random search to fine-tune the Decision Tree model
set.seed(321)
dt_model_rs <- train(Target ~ ., data = training_data, method = "rpart", trControl = train_control_rs, 
                     metric = "Accuracy", tuneLength = 15)


# Test the tuned model on training data and present the results with a confusion matrix
dt_predictions_rs <- predict(dt_model_rs, training_data)
confusionMatrix(dt_predictions_rs, training_data$Target, mode = "everything")

# Assess the performance of the tuned model on test data using a confusion matrix
dt_predictions_rs_test <- predict(dt_model_rs, testing_data)
confusionMatrix(dt_predictions_rs_test, testing_data$Target, mode = "everything")

# Implement grid search method for further hyperparameter optimization
train_control_grid <- trainControl(method = "cv", number = 10, search = "grid", verboseIter = TRUE, allowParallel = TRUE)
grid <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))

# Execute grid search to further fine-tune the Decision Tree model
set.seed(321)
dt_model_gs <- train(Target ~ ., data = training_data, method = "rpart", trControl = train_control_grid, 
                     metric = "Accuracy", tuneGrid = grid)

# Print the details of the model
print(dt_model_gs)


# Evaluate the grid-search optimized model on training data and display using a confusion matrix
dt_predictions_gs <- predict(dt_model_gs, training_data)
confusionMatrix(dt_predictions_gs, training_data$Target, mode = "everything")

# Test the model with test data and showcase results with a confusion matrix
dt_predictions_gs_test <- predict(dt_model_gs, testing_data)

# Present confusion matrix with detailed metrics
dt_metrics <- confusionMatrix(dt_predictions_gs_test, testing_data$Target, mode = "everything")
dt_metrics

# Measure model performance using Area Under the Curve

# Create a numeric vector of the true class levels
true_classes <- as.numeric(testing_data$Target) - 1  

# Predict probabilities for each class
dt_prob_predictions_test <- predict(dt_model_gs, testing_data, type = "prob")

# Calculate the AUC for each class using a one-vs-all approach
auc_values <- sapply(seq_along(dt_prob_predictions_test[1, ]), function(i) {
  roc_auc <- roc(response = ifelse(true_classes == (i-1), 1, 0), 
                 predictor = as.numeric(dt_prob_predictions_test[, i]))
  auc(roc_auc)
})

# Output the AUC values for each class
print(auc_values)

# Calculate the average AUC across all classes
avg_auc <- mean(auc_values)
print(avg_auc)

# Section 5: Validation of All Three Models 

# Extracting AUC values for each model
rf_auc_value <- auc(rf_auc)            # AUC for Random Forest
dt_auc_value <- avg_auc                # Average AUC for Decision Tree (after tuning)
xgb_auc_value <- auc(xgb_auc)          # AUC for XGBoost

# Assembling a data frame with model performance metrics
comparison_df <- data.frame(
  Model = c("Random Forest", "Decision Tree", "XGBoost"),
  Accuracy = c(rf_metrics$overall['Accuracy'], dt_metrics$overall['Accuracy'], xgb_metrics$overall['Accuracy']),
  AUC = c(rf_auc_value, dt_auc_value, xgb_auc_value)
)

# Displaying comparative model metrics
print(comparison_df)

# Reshape the data from wide to long format
long_comparison_df <- melt(comparison_df, id.vars = "Model", variable.name = "Metric", value.name = "Value")

# Creating a bar plot to compare model accuracies and AUC values with pastel colors
ggplot(data = long_comparison_df, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
  ylab("Metric Value") +
  xlab("Models") +
  ggtitle("Model Performance Comparison") +
  scale_fill_manual(values = c("Accuracy" = "#FADADD", "AUC" = "#FFA07A")) +
  theme_minimal()
