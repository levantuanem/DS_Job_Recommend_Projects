# DS_Job_Level_Recommend
A machine learning project that predicts job seniority levels from job posting data using NLP, TF-IDF, and classification models.
# DS Job Level Recommendation
Machine Learning project for predicting job experience levels from job posting information using Natural Language Processing (NLP) and classical Machine Learning models.
---
## 1. Project Overview
**DS Job Level Recommendation** is a Machine Learning project that predicts the experience level required for a job posting based on information such as:
* Job title
* Job description
* Required skills
* Salary information
* Other available job posting features
The project is designed following a modular Machine Learning architecture with separate components for data processing, feature engineering, model training, evaluation, visualization, configuration, and utilities.
### Main Objective
Build a reproducible Machine Learning pipeline that can:
1. Load raw job posting data
2. Clean and preprocess the dataset
3. Extract useful features from text and numerical data
4. Handle class imbalance
5. Train Machine Learning models
6. Evaluate model performance
7. Save trained models and preprocessing objects
8. Provide a reusable prediction pipeline
---
## 2. Problem Statement
Job postings often contain information about the required experience level of a candidate.
The objective of this project is to develop a classification model that predicts the required job level from job posting information.
### Target Classes
The project currently works with the following experience-level categories:
* Internship
* Entry level
* Associate
* Mid-Senior level
* Director
* Executive
This is formulated as a **multi-class classification problem**.
---
## 3. Dataset
The project uses a LinkedIn job postings dataset.
### Raw Dataset
```text
data/raw/postings.csv
```
### Processed Dataset
```text
data/processed/postings_preprocessed.csv
```
The dataset contains job posting information including text fields, salary information, company information, and experience-level labels.
---
## 4. Project Architecture
The project follows a modular Machine Learning architecture:
```text
                    ┌──────────────────┐
                    │   Raw Dataset    │
                    │   postings.csv   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Processing  │
                    │ Cleaning / Split │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature          │
                    │ Engineering     │
                    │ TF-IDF + Numeric │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Class Imbalance  │
                    │ Handling         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Model Training   │
                    │ Classification   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Evaluation       │
                    │ Metrics / Report │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Saved Model      │
                    │ + Preprocessor   │
                    └──────────────────┘
```
---
## 5. Repository Structure
```text
DS_Job_Level_Recommend/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── postings.csv
│   │
│   └── processed/
│       └── postings_preprocessed.csv
│
├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── reports/
│   └── ...
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── ...
│   │
│   └── utils/
│       ├── __init__.py
│       └── ...
│
├── tests/
│   └── ...
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```
---
## 6. Directory Responsibilities
| Directory            | Responsibility                           |
| -------------------- | ---------------------------------------- |
| `configs/`           | Project configuration files              |
| `data/raw/`          | Original datasets                        |
| `data/processed/`    | Cleaned and processed datasets           |
| `models/`            | Trained models and preprocessing objects |
| `notebooks/`         | Experiments and exploratory analysis     |
| `reports/`           | Generated reports and visualizations     |
| `src/data/`          | Data loading and preprocessing           |
| `src/features/`      | Feature engineering                      |
| `src/models/`        | Model training and evaluation            |
| `src/pipeline/`      | End-to-end ML pipeline                   |
| `src/visualization/` | EDA and visualization                    |
| `src/utils/`         | Reusable helper functions                |
| `tests/`             | Automated tests                          |
---
## 7. Tech Stack
### Programming Language
* Python 3.x
### Data Processing
* NumPy
* Pandas
### Machine Learning
* Scikit-learn
* Imbalanced-learn
### NLP
* TF-IDF
* Scikit-learn `TfidfVectorizer`
### Visualization
* Matplotlib
* Seaborn
### Configuration
* PyYAML
### Model Persistence
* Joblib
### Development Tools
* Git
* GitHub
* VS Code
---
## 8. Environment Setup

### 8.1 Clone Repository
```bash
git clone <repository-url>
```
Move into the project directory:
```bash
cd DS_Job_Level_Recommend
```
---
### 8.2 Create Virtual Environment
On Windows:
```bash
python -m venv .venv
```
Activate the environment:
```bash
.venv\Scripts\activate
```
After activation, the terminal should display something similar to:
```text
(.venv) D:\AI_Projects\DS_Job_Level_Recommend>
```
---
### 8.3 Install Dependencies
```bash
pip install -r requirements.txt
```
Verify the environment:
```bash
python --version
pip --version
```
---
## 9. Configuration
Project configuration is managed through:
```text
configs/config.yaml
```
Example:
```yaml
project:
  name: DS_Job_Level_Recommend
  random_state: 42

data:
  raw_path: data/raw/postings.csv
  processed_path: data/processed/postings_preprocessed.csv

model:
  output_dir: models

training:
  test_size: 0.2
  random_state: 42
```
Using a configuration file helps avoid hard-coded paths and parameters throughout the source code.
---
## 10. Data Pipeline
The data processing workflow is:

```text
Raw Data
   │
   ▼
Load Dataset
   │
   ▼
Remove Irrelevant Columns
   │
   ▼
Handle Missing Values
   │
   ▼
Remove Duplicates
   │
   ▼
Create Combined Text
   │
   ▼
Feature Engineering
   │
   ▼
Processed Dataset
```
Text information is combined from relevant fields such as:
```text
title
+
description
+
skills_desc
```
The resulting text representation is used for TF-IDF feature extraction.
---
## 11. Feature Engineering
The project uses both text and numerical features.
### Text Features
TF-IDF is used to transform job posting text into numerical feature vectors.
```python
TfidfVectorizer(stop_words="english")
```
### Numerical Features
Examples include:
* `min_salary`
* `med_salary`
* `max_salary`
* `normalized_salary`
The final feature representation combines textual and numerical information.
---
## 12. Class Imbalance
The target variable contains imbalanced classes.
To reduce the impact of class imbalance, the project can use:
```python
RandomOverSampler
```
Oversampling is applied only to the training data.

Correct workflow:

```text
Dataset
   │
   ▼
Train / Test Split
   │
   ├──────────────► Test Set
   │
   ▼
Training Set
   │
   ▼
Oversampling
   │
   ▼
Model Training
```
The test set remains untouched to provide an unbiased evaluation.
---
## 13. Model Training
The project is designed to support multiple classification models.
Potential models include:
* Logistic Regression
* Random Forest
* Gradient Boosting
* Linear SVM
* Other Scikit-learn classifiers
The model training workflow is:

```text
Processed Data
      │
      ▼
Train / Test Split
      │
      ▼
Feature Transformation
      │
      ▼
Class Balancing
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Save Model
```
---
## 14. Model Evaluation
The following classification metrics are used:
* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix
Example:
```text
Accuracy: 0.7655
```
The final model will be selected based on overall performance and suitability for the problem rather than accuracy alone.
---
## 15. Running the Project
### Current Development Workflow
The project is developed module by module.
### Data Processing
```bash
python -m src.data.preprocess
```
### Feature Engineering
```bash
python -m src.features.build_features
```
### Model Training
```bash
python -m src.models.train
```
### Model Evaluation
```bash
python -m src.models.evaluate
```
### End-to-End Pipeline
After the complete pipeline has been implemented:

```bash
python -m src.pipeline.run
```
The end-to-end pipeline should execute:

```text
Data
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Training
 ↓
Evaluation
 ↓
Model Saving
```
> Note: The commands above correspond to the planned modular architecture. They should only be used after the corresponding Python modules have been implemented.

---
## 16. Git Workflow
The project uses a feature-branch workflow.

```text
main
  │
  ▼
develop
  │
  ├── feature/config
  ├── feature/data
  ├── feature/model
  ├── feature/pipeline
  ├── feature/utils
  └── feature/visualization
```
### Main Branches
#### `main`
Stable production-ready version.
#### `develop`
Main development branch where completed features are integrated.
#### `feature/*`
Individual development branches for specific components.
---
## 17. Feature Development Workflow
Example: working on configuration.
Switch to the feature branch:
```bash
git switch feature/config
```
Check status:
```bash
git status
```
After making changes:
```bash
git add .
```
Commit:
```bash
git commit -m "Add project configuration"
```
Push:
```bash
git push -u origin feature/config
```
After the feature is completed, merge it into `develop`.
```text
feature/config
       │
       ▼
    develop
       │
       ▼
      main
```
---
## 18. Development Principles
The project follows several principles:
### Reproducibility
Important parameters should be stored in:
```text
configs/config.yaml
```
### Modularity
Each component should have a clear responsibility.
### Separation of Concerns
Data processing, feature engineering, model training, visualization, and utilities should remain separated.
### Version Control
Git branches are used to isolate feature development.
### Reusable Pipeline
The final system should be executable without manually running individual notebook cells.
---
## 19. Future Improvements
Planned improvements include:
* Hyperparameter optimization
* Better NLP feature extraction
* Transformer-based text representations
* Model comparison
* Cross-validation
* Improved class imbalance strategies
* Model explainability
* Automated testing
* End-to-end ML pipeline
* Model deployment
* API for job-level prediction
* Dockerization
* CI/CD integration
---
## 20. Project Status
Current development stages:
```text
[✓] Git repository setup
[✓] Main branch
[✓] Develop branch
[✓] Feature branch structure
[✓] Basic project structure
[ ] Configuration implementation
[ ] Data pipeline
[ ] Feature engineering pipeline
[ ] Model training pipeline
[ ] Evaluation pipeline
[ ] End-to-end pipeline
[ ] Automated tests
[ ] Deployment
```
---
## 21. License
This project is intended for educational, research, and portfolio purposes.
