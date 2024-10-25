# Machine Learning in Education: A Comparative Study of Predictive Models for Student Dropout

## 📚 Introduction

With advancements in machine learning and data mining technologies, predicting student academic performance, specifically identifying students at risk of dropping out, has become a growing area of research. Educational Data Mining (EDM) applies data mining techniques to academic data, uncovering patterns related to student retention, dropout rates, and overall academic success. 

By leveraging student demographics, academic performance, and socioeconomic data, predictive models offer valuable insights that can help institutions implement proactive interventions to improve student outcomes. The increasing availability of big data and advanced analytics further enhances the potential for accurate predictions and timely support. These predictive insights not only benefit individual students but also contribute to improving institutional performance and educational quality on a broader scale.


## 📋 Critical Discussion Points

### CHAPTER 1: OVERVIEW

#### 📌 1.1 Significance of Study

    ▪️ Predicting academic outcomes helps institutions identify struggling students early for timely intervention.
    
    ▪️ Insights from the prediction models can inform strategies to reduce dropout rates and improve retention.
    
    ▪️ Reducing dropout rates has broader societal benefits, including better employability and economic growth.

    ▪️ Effective resource allocation is ensured by targeting support to students most at risk.
    

#### 📌 1.2 Aim and Objectives 

Aim:

    Identify the best-performing model for predicting student dropouts through comprehensive hyperparameter tuning.
    

Objectives: 

    ▪️ Evaluate and determine the most reliable model for predicting dropout risk.
    
    ▪️ Identify key indicators that influence academic outcomes.
    
    ▪️ Demonstrate the impact of hyperparameter tuning on model performance.

    ▪️ Provide recommendations for reducing dropout rates based on findings.
    

#### 📌 1.3 Scope of the Study

    ▪️ Dataset features 4424 student records from 2008-2019.
    
    ▪️ Focus is on the fields of study represented in the dataset. 
    
    ▪️ Machine learning models used: XGBoost, Random Forest, and Decision Tree.
    
    ▪️ Evaluation metrics include accuracy, precision, recall, and AUC-ROC.

    ▪️ Analysis limited to the R programming language and RStudio tools.

---

### CHAPTER 2: RELATED WORK

#### 📌 2.1 Review of Student Dropout Prediction Models

    ▪️ Various studies have explored machine learning models for predicting student dropouts, with Random Forest frequently noted as a strong performer.
    
    ▪️ Timing of prediction is key; studies suggest optimal intervention should occur by the end of the first semester.
    
    ▪️ Features influencing dropout prediction differ by model, with socioeconomic, academic, and demographic factors commonly used.

    ▪️ Comparison studies indicate that boosting algorithms like XGBoost and LightGBM often outperform classical models like Random Forest and Decision Tree.


#### 📌 2.2 Research Gap

    ▪️ VLimited exploration of the impact of hyperparameter tuning in previous studies, which affects model optimization and performance.
    
    ▪️ Lack of detailed documentation in existing literature regarding the hyperparameter tuning process, making replication and further research difficult.
    
    ▪️ This study aims to address these gaps by conducting a comprehensive hyperparameter optimization process and providing detailed documentation for transparency and replication.

---

### CHAPTER 3: METHODS

#### 📌 3.1 Description of the Dataset

    ▪️ The dataset includes 4424 student records from 2008-2019, covering various fields like IT, business, and nursing.
    
    ▪️ Data is categorized into demographics, socioeconomics, macroeconomics, and academic performance, with a total of 37 attributes (including the target variable).
    
    ▪️ Data sources: Academic Management System, General Directorate of Higher Education, and the Contemporary Portugal Database.
    

#### 📌 3.2 Learning Techniques

##### 3.2.1 Programming Tools and Language

    ▪️ R programming language is used for its strong statistical analysis and data visualization capabilities.
    
    ▪️ RStudio serves as the Integrated Development Environment (IDE) for the analysis.
    

##### 3.2.2 Packages Utilized

    ▪️ Key R packages include reshape2, rpart, MLmetrics, xgboost, randomForest, ggplot2, and others for tasks like data manipulation, model implementation, and evaluation.
    

##### 3.2.3 Flowchart of Data Analysis Process

    ▪️ The data analysis process involves data preprocessing, exploratory data analysis (EDA), model implementation, and evaluation.
    

##### 3.2.4 Rationale for the Choice of Machine Learning Models

    ▪️ XGBoost, Random Forest, and Decision Tree models are chosen due to their effectiveness in handling missing data, overfitting, and providing interpretable results.
    

##### 3.2.5 Rationale for the Choice of Evaluation Metrics

    ▪️ Accuracy, Precision, Recall, and AUC-ROC metrics are selected to assess the model’s performance in predicting dropout risk.
    
    ▪️ A balance between Precision and Recall is emphasized, especially for the dropout class.

---

### CHAPTER 4: PREPARATION OF THE DATASET

#### 📌 4.1 Data Preprocessing

##### 4.1.1 Initial Data Exploration

    ▪️ A quick preview shows 37 attributes with a mix of numerical, categorical, binary, and discrete data types.
    
    ▪️ Initial observations include data type mismatches (e.g., gender and target variable incorrectly coded as numeric and character).

    ▪️ Identified wide ranges in certain attributes (e.g., GDP, admission grades) indicating potential need for normalization or scaling.
    
    ▪️ No missing values were detected, making preprocessing more straightforward.

    ▪️ Outliers were detected in some continuous variables (e.g., previous qualification grades), treated with mean adjustments for some columns.


##### 4.1.2 Renaming of the Column Label

    ▪️ Corrected a spelling error in the column label "Nacionality" to "Nationality" for clarity and consistency.


##### 4.1.3 Conversion of Variables into Factor Data

    ▪️ Converted attributes with mismatched data types (e.g., gender, target variable) into appropriate factor data types for accurate analysis and model compatibility.


##### 4.1.4 Detection and Treatment of Outliers

    ▪️ Identified outliers in continuous variables using the Interquartile Range (IQR) method, with a conservative IQR multiplier of 3.
    
    ▪️ Treated outliers in key variables (e.g., previous qualification grade and admission grade) by replacing extreme values with mean values, ensuring data integrity for modeling.


##### 4.1.5 Treatment of Problematic Entries

    ▪️ Zero values in academic attributes were identified and treated as missing values, and imputation was performed using the missForest package.


#### 📌 4.2 Exploratory Data Analysis (EDA)

##### 4.2.1 Analysis on the Target Column

    ▪️ Target class distribution was balanced enough, with no need for class balancing techniques.


##### 4.2.2 Correlation Matrix

    ▪️ A correlation heatmap was used to examine relationships between attributes.
    
    ▪️ Strong correlations were noted between curricular attributes (e.g., units approved) and the target variable, while parental qualifications showed minimal correlations.

    ▪️ Potential multicollinearity between independent variables was identified, with suggestions for feature reduction or regularization techniques if needed.
    
---

### CHAPTER 5: IMPLEMENTING THE MODEL 
 
#### 📌 5.1 Preparing the Data for Modelling

##### 5.1.1 Preparing the Target Variable

    ▪️ Target classes (“Graduate,” “Enrolled,” “Dropout”) were recoded numerically (0, 1, 2) for compatibility with the XGBoost model.


##### 5.1.2 Normalizing Numerical Attributes

    ▪️ Standardization was applied to numerical attributes to enhance model performance, especially for XGBoost.


##### 5.1.3 Splitting Data into Training and Test Sets

    ▪️ Data was split into 70% training and 30% testing sets using the caTools package for evaluation purposes.


#### 📌 5.2 Implementation of XGBoost Model

##### 5.2.1 XGBoost Baseline Model

    ▪️ The baseline XGBoost model achieved 80.11% accuracy on the test data, though a slight overfitting tendency was noted.


##### 5.2.2 Refined XGBoost Model

    ▪️ Hyperparameter tuning using Random Search led to a refined model with improved dropout prediction (AUC = 0.8922).
    
    ▪️ Significant features included second-semester curricular approvals and tuition fee status.


#### 📌 5.3 Implementation of Random Forest Model

##### 5.3.1 Random Forest Baseline Model

    ▪️ The baseline model attained 77.32% accuracy on test data, with signs of overfitting.


##### 5.3.2 Refined Random Forest Model:

    ▪️ Hyperparameter tuning (Random Search and Grid Search) improved accuracy to 79.95%.
    
    ▪️ Curricular units and semester grades were highly significant in predicting dropout.


#### 📌 5.4 Implementation of Decision Tree Model

##### 5.4.1 Decision Tree Baseline Model

    ▪️ The baseline Decision Tree model achieved 77.47% accuracy, with lower precision and recall for the “Enrolled” class.


 ##### 5.4.2 Refined Decision Tree Model

    ▪️ Hyperparameter tuning had minimal impact, with accuracy remaining at 77.47%.  
    
    ▪️ The model showed some limitations in further optimization.

---

### CHAPTER 6: MODEL EVALUATION
 
#### 📌 6.1 Comparison Across Implemented Models

    ▪️ XGBoost, Random Forest, and Decision Tree models were compared using key metrics: Accuracy, Precision, Recall, and AUC.
    
    ▪️ XGBoost achieved the highest overall accuracy and AUC, showing superior predictive power, especially for the dropout class.
    
    ▪️ Random Forest demonstrated solid performance but had some overfitting, particularly with training data.
    
    ▪️ Decision Tree showed lower accuracy and limited improvement despite tuning efforts, indicating constraints in model complexity.


#### 📌 6.2 Comparisons Among Peer Models

    ▪️ Results from this study were consistent with prior research, reaffirming that boosting algorithms (XGBoost) generally outperform classical models (Random Forest and Decision Tree) in dropout prediction.
Each model’s strengths and limitations were analyzed to understand their application in educational data mining.
    
    ▪️ Each model’s strengths and limitations were analyzed to understand their application in educational data mining.

 ---

### CHAPTER 7: ANALYSIS AND RECOMMENDATIONS
 
#### 📌 7.1 Critical Evaluations of Model Outcomes

    ▪️ XGBoost emerged as the most effective model for dropout prediction, with high accuracy and reliable performance across metrics.
    
    ▪️ Random Forest provided valuable insights but was prone to overfitting, requiring further refinement for optimal deployment.
    
    ▪️ Decision Tree was less effective due to limited model complexity and minimal improvement from hyperparameter tuning.
    

  #### 📌 7.2 Surprises or Anomalies

    ▪️ Some features, such as parental occupation and qualification, had minimal impact on dropout prediction, contrary to initial expectations.
    
    ▪️ Financial factors (e.g., tuition fee status) and academic performance (e.g., curricular unit approvals) were stronger predictors than demographic variables.
    

 #### 📌 7.3 Recommendations

    ▪️ Institutions should prioritize monitoring financial and academic indicators, as these are critical predictors of dropout risk.
    
    ▪️ Regular model refinement and hyperparameter tuning are recommended to enhance model accuracy and adaptability over time.

    ▪️ Implementing early warning systems based on these models can enable proactive interventions to reduce dropout rates.

---

### CHAPTER 8: CONCLUSIONS

    ▪️ The study identified XGBoost as the most effective model for predicting student dropout, demonstrating high accuracy and consistent performance across evaluation metrics.
    
    ▪️ Key predictors of dropout included academic performance indicators, such as curricular unit approvals, and financial factors like tuition fee status, allowing the model to pinpoint areas where institutions can focus support efforts to reduce dropout rates.

    ▪️ This study addressed gaps in prior research by implementing comprehensive hyperparameter tuning and providing detailed documentation for replicability.

    ▪️ Findings suggest that machine learning models can help institutions identify at-risk students early, supporting timely interventions that improve student retention and success rates.

    ▪️ Future work could involve further hyperparameter tuning, model refinement through ensemble techniques, and additional feature engineering to uncover deeper predictive insights.

















