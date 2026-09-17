# DS Job Recommend Projects
## Job Level Analysis and Prediction Based on Job Posting Content and Features
---

## 1. Project Overview
**DS Job Recommend Projects** is a Data Science and Machine Learning project that analyzes LinkedIn Job Postings to understand the characteristics of job levels and build a machine learning system for **Job Level Prediction**.

- The project focuses on combining:
  * Data Understanding & Data Cleaning
  * Exploratory Data Analysis (EDA)
  * Statistical Analysis
  * Feature Engineering
  * Natural Language Processing (NLP)
  * Skill Extraction
  * Machine Learning
  * Model Evaluation & Interpretation
  * End-to-End Pipeline
  * Business Insights

### Main Target
The primary prediction target is:
```text
formatted_experience_level
```
Example classes:
```text
Entry Level
Associate
Mid-Senior Level
Director
Executive
Internship
```
---
# 2. Project Objectives
The project aims to:
1. Understand the structure and quality of LinkedIn job posting data.
2. Analyze characteristics of different job levels.
3. Analyze relationships between features.
4. Analyze relationships between features and job level.
5. Identify skills and requirements associated with each job level.
6. Perform numerical, categorical, and text feature engineering.
7. Apply NLP techniques to job posting content.
8. Extract skills from job descriptions.
9. Build Machine Learning models for job level prediction.
10. Analyze loss functions and model optimization.
11. Analyze bias and variance.
12. Detect underfitting and overfitting.
13. Compare different Machine Learning models.
14. Interpret model decisions and identify important features.
15. Build a reproducible end-to-end pipeline.
16. Generate meaningful business insights.
---

# 3. Data Science Workflow
```text
Business Understanding
        ↓
Data Collection
        ↓
Data Understanding
        ↓
Data Quality Analysis
        ↓
Data Cleaning
        ↓
EDA
        ↓
Feature ↔ Feature Analysis
        ↓
Feature ↔ Target Analysis
        ↓
Statistical Analysis
        ↓
Feature Engineering
        ↓
NLP / Text Processing
        ↓
Skill Extraction
        ↓
Feature Selection
        ↓
Preprocessing
        ↓
Modeling
        ↓
Cross Validation
        ↓
Hyperparameter Tuning
        ↓
Evaluation
        ↓
Model Interpretation
        ↓
Prediction
        ↓
Business Insights
```
---

# 4. Project Architecture
```text
DS_Job_Recommend_Projects/
│
├── .vscode/
│
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   ├── model.yaml
│   └── pipeline.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── scripts/
│   ├── train.py
│   └── predict.py
│
├── src/
│   ├── data/
│   ├── features/
│   ├── visualization/
│   ├── models/
│   ├── pipeline/
│   └── utils/
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── README.md
└── requirements.txt
```
### Folder Responsibilities
| Folder               | Purpose                                 |
| -------------------- | --------------------------------------- |
| `.vscode/`           | VS Code project configuration           |
| `configs/`           | Project configuration and parameters    |
| `data/raw/`          | Original dataset                        |
| `data/interim/`      | Intermediate data                       |
| `data/processed/`    | Cleaned and processed data              |
| `models/`            | Trained models and reusable artifacts   |
| `notebooks/`         | Exploration, EDA and experiments        |
| `scripts/`           | Standalone execution scripts            |
| `src/data/`          | Data validation and cleaning            |
| `src/features/`      | Feature Engineering and NLP             |
| `src/visualization/` | Visualization functions                 |
| `src/models/`        | Training, evaluation and interpretation |
| `src/pipeline/`      | End-to-end workflow                     |
| `src/utils/`         | Shared utilities                        |
| `tests/`             | Unit and integration tests              |
---

# 5. Team Responsibilities
The project is divided into five main responsibilities.
```text
develop
│
├── feature/data
│       └── Member 1
│
├── feature/visualization
│       └── Member 2
│
├── feature/features
│       └── Member 3
│
├── feature/model
│       └── Member 4
│
└── feature/pipeline
        └── Member 5
```
---

# 6. Member 1 — Data Collection & Data Cleaning
### Branch
```text
feature/data
```
### Role
**Data Engineer / Data Analyst**
### Responsibilities
  * Data Collection
  * Data Understanding
  * Data Quality Analysis
  * Data Cleaning
  * Dataset Documentation

### Main Tasks
#### Data Collection
* Verify dataset source.
* Store original data in `data/raw/`.
* Never directly modify raw data.
* Record dataset information:

  * Source
  * Number of records
  * Number of features
  * Dataset scope

#### Data Understanding
Analyze:
  * Shape
  * Column names
  * Data types
  * Missing values
  * Unique values
  * Cardinality
  * Invalid values
  * Numerical and categorical features
  * Text features
Important feature groups include:

```text
JOB INFORMATION
├── title
├── description
├── skills_desc
└── formatted_experience_level

SALARY
├── min_salary
├── med_salary
├── max_salary
├── normalized_salary
├── currency
├── pay_period
└── compensation_type

LOCATION
├── location
├── zip_code
└── fips

WORK
├── formatted_work_type
├── work_type
└── remote_allowed

ENGAGEMENT
├── views
└── applies

TIME
├── listed_time
├── original_listed_time
├── expiry
└── closed_time

COMPANY
└── company information
```

#### Data Quality
Check:
  * Missing values
  * Duplicate records
  * Incorrect data types
  * Invalid categories
  * Inconsistent categories
  * Invalid text
  * Negative or impossible values
  * Extreme values
Each issue should document:

```text
Problem
Affected Records
Affected Feature
Impact
Treatment
Reason
```

#### Output
```text
Clean Dataset
Data Quality Report
Data Understanding Notebook
Data Cleaning Notebook
Reusable Cleaning Functions
Dataset Documentation
```

Main folders:
```text
data/raw/
data/processed/
src/data/
tests/test_data.py
notebooks/
```
---

# 7. Member 2 — EDA, Statistics & Visualization
### Branch
```text
feature/visualization
```
### Role
**Data Analyst / Visualization Analyst**
### Responsibilities
  * Exploratory Data Analysis
  * Statistical Analysis
  * Feature ↔ Feature Analysis
  * Feature ↔ Target Analysis
  * Visualization
  * Skill Analysis
  * Text Analysis
  * Business Insights

---

## 7.1 Univariate Analysis
### Numerical Features
Analyze:
  * Salary
  * Views
  * Applies
  * Normalized Salary
  * Other numerical features

Statistics:
  * Mean
  * Median
  * Standard Deviation
  * Min / Max
  * Quartiles
  * IQR
  * Skewness
  * Distribution
  * Outliers

Visualizations:
  * Histogram
  * KDE
  * Boxplot

### Categorical Features
Analyze:
  * Job Level
  * Work Type
  * Employment Type
  * Location
  * Remote
  * Pay Period

Analyze:
  * Frequency
  * Percentage
  * Cardinality
  * Class imbalance

Visualizations:
  * Bar Chart
  * Count Plot
  * Percentage Chart
---

## 7.2 Feature ↔ Feature Analysis
Analyze relationships between features.
### Numerical ↔ Numerical
Examples:
```text
Salary ↔ Views
Salary ↔ Applies
Views ↔ Applies
```

Methods:
* Pearson Correlation
* Spearman Correlation
* Correlation Matrix
* Heatmap
* Scatter Plot

Goals:
* Identify correlation
* Identify redundancy
* Detect multicollinearity
* Identify highly related features
---

## 7.3 Feature ↔ Target Analysis
Target:
```text
formatted_experience_level
```

Analyze:
```text
Numerical Feature ↔ Job Level
Categorical Feature ↔ Job Level
Text Feature ↔ Job Level
```

Examples:
```text
Salary ↔ Job Level
Views ↔ Job Level
Applies ↔ Job Level
Work Type ↔ Job Level
Remote ↔ Job Level
Employment Type ↔ Job Level
```

Text-derived analysis:
```text
Description Length
Title Length
Skill Count
Keyword Frequency
```

Possible statistical methods:
  * ANOVA
  * Kruskal-Wallis
  * Chi-square
  * Effect Size
  * Group Comparison

> Correlation or association does not imply causation.

---

## 7.4 Categorical ↔ Numerical Analysis
Examples:
```text
Job Level ↔ Salary
Work Type ↔ Salary
Remote ↔ Salary
Job Level ↔ Views
Job Level ↔ Applies
```

Visualizations:
* Boxplot
* Violin Plot
* Bar Chart
* Grouped Bar Chart
---

## 7.5 Categorical ↔ Categorical Analysis
Examples:
```text
Job Level ↔ Work Type
Job Level ↔ Remote
Job Level ↔ Employment Type
Job Level ↔ Location
```

Methods:
* Crosstab
* Percentage
* Chi-square test

Visualizations:
* Stacked Bar Chart
* Grouped Bar Chart
---

## 7.6 Multivariate Analysis
Analyze:
* Multiple numerical features
* Feature interactions
* Multicollinearity
* Multiple features versus target
* Differences between job levels
---

## 7.7 Outlier Analysis
Analyze:
```text
Salary
Views
Applies
Other numerical features
```

Possible methods:
* IQR
* Z-score
* Log Transformation

Outliers should not automatically be removed.
The team must distinguish between:
```text
Data Error
       vs
Genuine Extreme Value
```
---

## 7.8 Class Imbalance
Analyze the distribution of:
```text
Entry Level
Associate
Mid-Senior Level
Director
Executive
Internship
```

Results are passed to **Member 4** for modeling.
---

## 7.9 Skill Analysis
Analyze:
* Most common skills
* Skills by job level
* Skills by location
* Skills by work type
* Skills associated with senior positions
* Skills associated with entry-level positions

Sources:
```text
title
description
skills_desc
```

Important distinction:
```text
Skill Analysis
→ Business / EDA

Skill Extraction
→ Machine Learning Features
```
---

## 7.10 Text Analysis
Analyze:
* Title length
* Description length
* Word count
* Skill count
* Keyword frequency
* Text length by job level
* Keywords by job level

Text Analysis describes the data.
NLP converts text into Machine Learning features.
---

## 7.11 Business Insights
Answer questions such as:
* Which job level appears most frequently?
* Which job level has the highest salary?
* Is salary associated with job level?
* Which features are strongly correlated?
* Is multicollinearity present?
* Which features are associated with job level?
* Which skills are most common?
* Which skills characterize each job level?
* Which locations have the most jobs?
* Which locations have higher salaries?
* Which job levels contain more remote positions?f
---

# 8. Member 3 — Feature Engineering & NLP

### Branch

```text
feature/features
```

### Role

**Feature Engineer / NLP Engineer**

### Responsibilities

* Numerical Feature Engineering
* Categorical Feature Engineering
* Text Feature Engineering
* NLP
* Skill Extraction
* Feature Selection
* Preprocessing
* Data Leakage Prevention

---

## 8.1 Numerical Feature Engineering

Possible features:

```text
salary_range
log_salary
application_rate
view_to_apply_ratio
```

Potential source features:

```text
min_salary
med_salary
max_salary
normalized_salary
views
applies
```

Special attention must be given to possible **data leakage** involving:

```text
views
applies
```

---

## 8.2 Categorical Feature Engineering

Possible techniques:

* One-Hot Encoding
* Frequency Encoding
* Ordinal Encoding where appropriate

Potential features:

```text
work_type
formatted_work_type
remote_allowed
pay_period
currency
location
application_type
```

---

## 8.3 Text Feature Engineering

Sources:

```text
title
description
skills_desc
```

Possible features:

```text
text_length
word_count
sentence_count
skill_count
keyword_count
technical_keyword_count
management_keyword_count
leadership_keyword_count
```

---

## 8.4 NLP

NLP workflow may include:

```text
Text Cleaning
      ↓
Lowercasing
      ↓
Noise Removal
      ↓
Tokenization
      ↓
Stopword Removal
      ↓
N-gram
      ↓
TF-IDF
```

> TF-IDF is one NLP technique. NLP is not limited to TF-IDF.

---

## 8.5 Skill Extraction

Extract skills from:

```text
title
description
skills_desc
```

Examples:

```text
has_python
has_sql
has_java
has_aws
has_excel
has_machine_learning
has_cloud
has_leadership
has_management
```

These extracted skills become Machine Learning features.

---

## 8.6 Feature Selection

Use EDA and statistical analysis to identify:

* Useful features
* Redundant features
* Highly correlated features
* Multicollinearity
* Leakage-prone features

Possible techniques:

* Correlation
* Variance
* Mutual Information
* SelectKBest
* Feature Importance

---

## 8.7 Data Leakage Prevention

Check:

```text
Target Leakage
Train/Test Leakage
Information Leakage
Temporal Leakage
```

Special attention:

```text
views
applies
```

Features should only be used if they would realistically be available at prediction time.

---

## 8.8 Preprocessing

Implement:

* Scaling
* Encoding
* Imputation
* Text Vectorization

Training and test data must use the same preprocessing logic.

However:

```text
Fit → Training Data only

Transform → Training / Validation / Test
```

---

# 9. Member 4 — Modeling & Evaluation

### Branch

```text
feature/model
```

### Role

**Machine Learning Engineer**

### Responsibilities

* Modeling
* Loss Function
* Class Imbalance
* Cross Validation
* Hyperparameter Tuning
* Bias / Variance
* Underfitting / Overfitting
* Evaluation
* Model Interpretation

---

## 9.1 Baseline Model

Start with:

```text
Logistic Regression
```

Purpose:

* Establish baseline performance.
* Provide a reference point for more complex models.

---

## 9.2 Candidate Models

Potential models:

```text
Logistic Regression
Random Forest
Linear SVM
Gradient Boosting
XGBoost
LightGBM
```

Models should be compared using the same evaluation strategy.

---

## 9.3 Loss Function

For multiclass classification:

```text
Cross-Entropy Loss
Log Loss
Weighted Cross-Entropy
```

Analyze:

* What the loss represents
* How the loss is optimized
* Training loss
* Validation loss where applicable
* Relationship between loss and overfitting

---

## 9.4 Class Imbalance

Possible approaches:

```text
class_weight
Random Over Sampling
SMOTE
```

Sampling must only be applied to the training data.

---

## 9.5 Cross Validation

Use Cross Validation to evaluate:

* Model stability
* Variance
* Generalization
* Hyperparameter configurations

Record:

```text
CV Mean
CV Standard Deviation
Scores by Fold
```

---

## 9.6 Hyperparameter Tuning

Possible methods:

```text
GridSearchCV
RandomizedSearchCV
```

Potential parameters:

* Regularization
* Tree Depth
* Number of Estimators
* Learning Rate
* Model-specific parameters

---

## 9.7 Bias Analysis

Compare:

```text
Training Performance
Validation Performance
Test Performance
```

High Bias typically appears when:

```text
Training Performance → Low
Validation Performance → Low
Test Performance → Low
```

This is commonly associated with **underfitting**.

---

## 9.8 Variance Analysis

High variance may appear when:

```text
Training Performance → Very High
Validation Performance → Low
Test Performance → Low
```

For iterative models, another indicator is:

```text
Training Loss → Low
Validation Loss → High
```

This is commonly associated with **overfitting**.

---

## 9.9 Evaluation

Do not rely only on Accuracy.

Evaluate:

```text
Accuracy
Precision
Recall
F1-score
Macro-F1
Weighted-F1
Confusion Matrix
```

When class imbalance is significant:

> **Macro-F1 should receive particular attention.**

---

## 9.10 Model Comparison

Create a comparison table:

| Model               | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 | Training Time | CV Mean | CV Std |
| ------------------- | -------: | --------: | -----: | -------: | ----------: | ------------: | ------: | -----: |
| Logistic Regression |          |           |        |          |             |               |         |        |
| Random Forest       |          |           |        |          |             |               |         |        |
| Linear SVM          |          |           |        |          |             |               |         |        |
| Gradient Boosting   |          |           |        |          |             |               |         |        |

The final model should be selected based on the project's evaluation criteria rather than a single metric.

---

## 9.11 Model Interpretation

Answer:

> Which features does the model use to determine Job Level?

Possible techniques:

* Feature Importance
* Permutation Importance
* SHAP
* Coefficient Analysis

Analyze:

```text
Numerical Features
Categorical Features
Skills
Keywords
Text Features
```

---

## 9.12 Model Artifacts

Save reusable artifacts such as:

```text
Best Model
Vectorizer
Encoder
Scaler
Label Mapping
```

These artifacts must support the prediction workflow.

---

# 10. Member 5 — Integration, Pipeline & Documentation

### Branch

```text
feature/pipeline
```

### Role

**Data Science / ML Engineer / Project Integration**

### Responsibilities

* Integration
* End-to-End Pipeline
* Configuration
* Utilities
* Testing
* Experiment Management
* Prediction
* Documentation
* Git Integration

---

## 10.1 Integration Workflow

Connect:

```text
Data
 ↓
Cleaning
 ↓
EDA
 ↓
Feature Engineering
 ↓
NLP
 ↓
Preprocessing
 ↓
Modeling
 ↓
Evaluation
 ↓
Prediction
```

---

## 10.2 End-to-End Pipeline

```text
Raw Dataset
      ↓
Validation
      ↓
Cleaning
      ↓
Feature Engineering
      ↓
NLP
      ↓
Preprocessing
      ↓
Train/Test Split
      ↓
Training
      ↓
Evaluation
      ↓
Save Model
      ↓
Prediction
```

---

## 10.3 Configuration

Configuration files:

```text
configs/
├── data.yaml
├── features.yaml
├── model.yaml
└── pipeline.yaml
```

### `data.yaml`

Contains:

* Data paths
* Cleaning parameters
* Split parameters

### `features.yaml`

Contains:

* Feature parameters
* NLP parameters
* TF-IDF parameters

### `model.yaml`

Contains:

* Model parameters
* Hyperparameters
* Evaluation parameters

### `pipeline.yaml`

Contains:

* Pipeline settings
* Output paths
* Experiment settings

---

## 10.4 Utilities

```text
src/utils/
├── config_loader.py
├── logger.py
├── random_seed.py
└── file_utils.py
```

These modules contain reusable project-wide utilities.

---

## 10.5 Testing

Tests:

```text
tests/
├──
```
