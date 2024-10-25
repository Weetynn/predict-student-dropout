# Machine Learning in Education: A Comparative Study of Predictive Models for Student Dropout

## 📚 Introduction

With advancements in machine learning and data mining technologies, predicting student academic performance, specifically identifying students at risk of dropping out, has become a growing area of research. Educational Data Mining (EDM) applies data mining techniques to academic data, uncovering patterns related to student retention, dropout rates, and overall academic success. 

By leveraging student demographics, academic performance, and socioeconomic data, predictive models offer valuable insights that can help institutions implement proactive interventions to improve student outcomes. The increasing availability of big data and advanced analytics further enhances the potential for accurate predictions and timely support. These predictive insights not only benefit individual students but also contribute to improving institutional performance and educational quality on a broader scale.


## 📋 Critical Discussion Points

### CHAPTER 1: INTRODUCTION

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

#### 📌 2.1 Review of Student Dropout Prediction Models

    ▪️ Various studies have explored machine learning models for predicting student dropouts, with Random Forest frequently noted as a strong performer.
    
    ▪️ Timing of prediction is key; studies suggest optimal intervention should occur by the end of the first semester.
    
    ▪️ Features influencing dropout prediction differ by model, with socioeconomic, academic, and demographic factors commonly used.

    ▪️ Comparison studies indicate that boosting algorithms like XGBoost and LightGBM often outperform classical models like Random Forest and Decision Tree.














