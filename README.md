<!-- conda activate ai_projects
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
This project is intended for educational, research, and portfolio purposes. -->

============================================================
TOPIC: Job Level Analysis and Prediction based on 
Job Posting Content and Features
============================================================
============================================================
PHÂN CHIA NHIỆM VỤ PROJECT DS_JOB_RECOMMEND_PROJECTS
============================================================
MỤC TIÊU PROJECT
------------------------------------------------------------
Phân tích dữ liệu tuyển dụng từ LinkedIn Job Postings để:
1. Hiểu đặc điểm của thị trường tuyển dụng.
2. Phân tích các yếu tố liên quan đến Job Level.
3. Phân tích các kỹ năng và yêu cầu trong từng Job Level.
4. Xây dựng Feature Engineering từ dữ liệu số, categorical và text.
5. Sử dụng Machine Learning để dự đoán Job Level.
6. Đánh giá và giải thích Model.
7. Đưa ra Business Insights từ dữ liệu và Model.
============================================================
DATA SCIENCE WORKFLOW
============================================================
Business Understanding
        ↓
Data Collection
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
EDA
        ↓
Statistical / Relationship Analysis
        ↓
Feature Engineering
        ↓
NLP / Text Analysis
        ↓
Modeling
        ↓
Evaluation
        ↓
Model Interpretation
        ↓
Prediction / Recommendation
============================================================
NGƯỜI 1 — DATA ANALYSIS
============================================================
- Branch: feature/data
- VAI TRÒ: Data Analyst
- MỤC TIÊU: Hiểu dữ liệu, làm sạch dữ liệu, phân tích dữ liệu và tìm ra
các mối quan hệ / insight quan trọng trước khi đưa dữ liệu
cho Feature Engineering và Modeling.
------------------------------------------------------------
1. DATA COLLECTION
------------------------------------------------------------
- Kiểm tra nguồn dataset.
- Đưa dataset gốc vào: data/raw/
- Không chỉnh sửa trực tiếp dữ liệu raw.
------------------------------------------------------------
2. DATA UNDERSTANDING
------------------------------------------------------------
- Kiểm tra:
   - Shape của dataset.
   - Số dòng / số cột.
   - Tên các feature.
   - Data type.
   - Missing values.
   - Unique values.
   - Cardinality.
   - Các giá trị bất thường.
- Phân loại feature:
   - JOB INFORMATION:
      - title
      - description
      - skills_desc
      - formatted_experience_level
   - SALARY:
      - min_salary
      - med_salary
      - max_salary
      - normalized_salary
      - currency
      - pay_period
      - compensation_type
   - LOCATION:
      - location
      - zip_code
      - fips
   - WORK:
      - formatted_work_type
      - work_type
      - remote_allowed
   - ENGAGEMENT:
      - views
      - applies
   - TIME:
      - listed_time
      - original_listed_time
      - expiry
      - closed_time
   - COMPANY:
      - company information
------------------------------------------------------------
3. DATA CLEANING
------------------------------------------------------------
- Xử lý:
   - Missing values.
   - Duplicate.
   - Sai datatype.
   - Giá trị không hợp lệ.
   - Category không đồng nhất.
   - Text bị lỗi.
   - Outlier.
   - Các giá trị không hợp lý.
- Phải ghi rõ:
   - Vấn đề là gì?
   - Có bao nhiêu record bị ảnh hưởng?
   - Cách xử lý?
   - Tại sao chọn cách xử lý đó?
- Output: data/processed/
------------------------------------------------------------
4. UNIVARIATE ANALYSIS
------------------------------------------------------------
Phân tích từng biến riêng lẻ.
- NUMERICAL:
   + Salary.
   + Views.
   + Applies.
   + Normalized salary.
   + Các numerical feature khác.
   - Phân tích:
      - Mean.
      - Median.
      - Standard deviation.
      - Min / Max.
      - Quartile.
      - IQR.
      - Distribution.
      - Skewness.
      - Outlier.
   - Visualization:
      - Histogram.
      - KDE.
      - Boxplot.
- CATEGORICAL:
   + Job Level.
   + Work Type.
   + Employment Type.
   + Location.
   + Remote.
   + Pay Period.
   - Phân tích:
      - Frequency.
      - Percentage.
      - Cardinality.
      - Category imbalance.
   - Visualization:
      - Histogram.
      - KDE.
      - Boxplot.
------------------------------------------------------------
5. BIVARIATE ANALYSIS
------------------------------------------------------------
Phân tích mối quan hệ giữa 2 biến.
- NUMERICAL ↔ NUMERICAL:
   - Phân tích:
      - Correlation.
      - Pearson correlation.
      - Spearman correlation.
      - Linear relationship.
      - Positive / Negative relationship.
      - Strength of relationship.
   - Visualization:
      - Scatter plot.
      - Regression plot.
      - Correlation plot.
- CATEGORICAL ↔ NUMERICAL:
   - Phân tích:
      - Mean.
      - Median.
      - Distribution.
      - Difference giữa các nhóm.
   - Visualization:
      - Boxplot.
      - Violin plot.
      - Bar chart.
- CATEGORICAL ↔ CATEGORICAL:
   - Phân tích:
      - Crosstab.
      - Percentage.
      - Chi-square test nếu phù hợp.
   - Visualization:
      - Boxplot.
      - Violin plot.
      - Bar chart.
------------------------------------------------------------
6. MULTIVARIATE ANALYSIS
------------------------------------------------------------
- Phân tích nhiều biến cùng lúc.
   - Phân tích:
      - Interaction giữa các feature.
      - Correlation giữa nhiều numerical feature.
      - Multicollinearity.
      - Feature relationship.
   - Visualization:
      - Correlation heatmap.
      - Pairplot nếu dataset phù hợp.
      - Grouped visualization.
------------------------------------------------------------
7. OUTLIER ANALYSIS
------------------------------------------------------------
- Phân tích outlier trong:
   - Salary.
   - Views.
   - Applies.
   - Các numerical feature khác.
- Xác định:
   - Outlier thật hay lỗi dữ liệu?
   - Có nên remove không?
   - Có nên transform không?
- Có thể sử dụng:
   - IQR.
   - Z-score.
   - Log transformation.
------------------------------------------------------------
8. CLASS IMBALANCE ANALYSIS
------------------------------------------------------------
Target: formatted_experience_level
- Kiểm tra:
   - Entry Level.
   - Associate.
   - Mid-Senior Level.
   - Director.
   - Executive.
   - Internship.
- Phân tích:
   - Số lượng từng class.
   - Tỷ lệ từng class.
   - Class imbalance.
- Kết quả được bàn giao cho người Modeling để xử lý.
------------------------------------------------------------
9. SKILL ANALYSIS
------------------------------------------------------------
- Phân tích:
   - Skill phổ biến nhất.
   - Skill theo Job Level.
   - Skill theo Location.
   - Skill theo Work Type.
   - Skill nào xuất hiện nhiều ở Senior?
   - Skill nào xuất hiện nhiều ở Director?
   - Skill nào đặc trưng cho Entry Level?
- Nguồn:
   - skills_desc
   - description
   - title
------------------------------------------------------------
10. TEXT ANALYSIS
------------------------------------------------------------
- Phân tích:
   - Độ dài title.
   - Độ dài description.
   - Word count.
   - Skill count.
   - Text length theo Job Level.
   - Keyword frequency.
   - Keyword theo Job Level.
------------------------------------------------------------
11. BUSINESS INSIGHTS
------------------------------------------------------------
- Phải rút ra kết luận từ EDA.
   - Job Level nào tuyển nhiều nhất?
   - Job Level nào có salary cao nhất?
   - Salary có quan hệ với Experience không?
   - Salary có quan hệ tuyến tính với Experience không?
   - Skill nào phổ biến nhất?
   - Skill nào đặc trưng cho từng Job Level?
   - Location nào tuyển nhiều?
   - Location nào có salary cao?
   - Remote job tập trung ở Job Level nào?
   - Có sự khác biệt rõ giữa các Job Level không?
- OUTPUT:
   - Clean dataset.
   - EDA notebooks.
   - Visualization.
   - Statistical analysis.
   - Business insights.
- Folder liên quan:
   data/
   notebooks/
   src/data/
   src/visualization/
============================================================
NGƯỜI 2 — FEATURE ENGINEERING + NLP
============================================================
- Branch đề xuất: feature/features
- VAI TRÒ: Feature Engineer / NLP Engineer
- MỤC TIÊU: Chuyển dữ liệu đã clean thành các feature có thể sử dụng
cho Machine Learning.
------------------------------------------------------------
1. NUMERICAL FEATURE ENGINEERING
------------------------------------------------------------
- Xử lý:
   - min_salary
   - med_salary
   - max_salary
   - normalized_salary
   - views
   - applies
- Có thể tạo:
   - salary_range
   - log_salary
   - application_rate
   - view_to_apply_ratio
- Phải kiểm tra Data Leakage trước khi sử dụng views / applies.
------------------------------------------------------------
2. CATEGORICAL FEATURE ENGINEERING
------------------------------------------------------------
- Xử lý:
   - work_type
   - formatted_work_type
   - remote_allowed
   - pay_period
   - currency
   - location
   - application_type
- Có thể sử dụng:
   - One-Hot Encoding.
   - Frequency Encoding.
   - Ordinal Encoding nếu phù hợp.
   - Các encoding khác nếu cần.
------------------------------------------------------------
3. TEXT FEATURE ENGINEERING
------------------------------------------------------------
- Nguồn:
   - title
   - description
   - skills_desc
- Tạo:
   - text_length
   - word_count
   - sentence_count
   - skill_count
   - keyword_count
   - technical_keyword_count
   - management_keyword_count
   - leadership_keyword_count
------------------------------------------------------------
4. NLP
------------------------------------------------------------
- Thực hiện:
   - Text cleaning.
   - Lowercase.
   - Remove noise.
   - Tokenization.
   - Stopword removal nếu phù hợp.
   - N-gram.
   - TF-IDF.
- LƯU Ý:
   TF-IDF chỉ là một phương pháp NLP.
   Không được coi: NLP = TF-IDF
------------------------------------------------------------
5. SKILL EXTRACTION
------------------------------------------------------------
Xác định các skill quan trọng.
Ví dụ:
   - Python.
   - SQL.
   - Java.
   - AWS.
   - Machine Learning.
   - Excel.
   - Cloud.
   - Leadership.
   - Management.
Có thể tạo:
   has_python
   has_sql
   has_aws
   has_java
   has_machine_learning
...
------------------------------------------------------------
6. FEATURE SELECTION
------------------------------------------------------------
- Kiểm tra:
   - Feature nào hữu ích?
   - Feature nào dư thừa?
   - Feature nào có correlation quá cao?
   - Feature nào gây multicollinearity?
   - Feature nào gây data leakage?
- Có thể sử dụng:
   - Correlation.
   - Variance.
   - Feature importance.
   - Mutual information.
   - SelectKBest nếu phù hợp.
------------------------------------------------------------
7. PREPROCESSING
------------------------------------------------------------
- Xử lý:
   - Scaling.
   - Encoding.
   - Imputation nếu cần.
   - Text vectorization.
- Output: X, y => sẵn sàng cho Modeling.
- Folder:
   src/features/
   configs/features.yaml
   tests/test_features.py
============================================================
NGƯỜI 3 — MODELING + EVALUATION
============================================================
- Branch: feature/model
- VAI TRÒ: Machine Learning Engineer
- MỤC TIÊU: Xây dựng, tối ưu, đánh giá và giải thích Machine Learning Model.
------------------------------------------------------------
1. TARGET
------------------------------------------------------------
- Target: formatted_experience_level
- Các class:
   - Entry Level
   - Associate
   - Mid-Senior Level
   - Director
   - Executive
   - Internship
------------------------------------------------------------
2. BASELINE MODEL
------------------------------------------------------------
- Thử các model:
   - Logistic Regression.
   - Random Forest.
   - Linear SVM.
   - Gradient Boosting.
   - XGBoost / LightGBM nếu phù hợp.
- Không chọn model chỉ vì nó phổ biến.
------------------------------------------------------------
3. TRAINING
------------------------------------------------------------
   - Train/Test Split.
   - Cross Validation.
   - Training.
   - So sánh model.
------------------------------------------------------------
4. CLASS IMBALANCE
------------------------------------------------------------
- Dựa trên kết quả phân tích của Người 1.
- Có thể thử:
   - class_weight.
   - Random Over Sampling.
   - SMOTE nếu phù hợp.
- Phải đảm bảo không xảy ra Data Leakage.
------------------------------------------------------------
5. HYPERPARAMETER TUNING
------------------------------------------------------------
- Sử dụng:
   - GridSearchCV.
   - RandomizedSearchCV.
- Tối ưu:
   - Model parameters.
   - Regularization.
   - Tree depth.
   - Number of estimators.
   - Learning rate.
   - Các parameter phù hợp.
------------------------------------------------------------
6. BIAS / VARIANCE
------------------------------------------------------------
- Phân tích:
   - Bias.
   - Variance.
   - Underfitting.
   - Overfitting.
- So sánh:
   Training performance
   vs
   Validation performance
   vs
   Test performance.
------------------------------------------------------------
7. EVALUATION
------------------------------------------------------------
- Không chỉ dùng Accuracy.
- Đánh giá:
   - Accuracy.
   - Precision.
   - Recall.
   - F1-score.
   - Macro-F1.
   - Weighted-F1.
   - Confusion Matrix.
- Nếu class imbalance cao: Ưu tiên Macro-F1.
------------------------------------------------------------
8. MODEL INTERPRETATION
------------------------------------------------------------
- Trả lời câu hỏi: "Model dựa vào những đặc điểm nào để xác định Job Level?"
- Có thể sử dụng:
   - Feature Importance.
   - Permutation Importance.
   - SHAP.
   - Coefficient analysis.
- Phân tích:
   - Feature quan trọng nhất.
   - Skill quan trọng nhất.
   - Keyword quan trọng nhất.
   - Numerical feature quan trọng.
   - Categorical feature quan trọng.
------------------------------------------------------------
9. MODEL ARTIFACT
------------------------------------------------------------
- Lưu:
   - Best model.
   - Vectorizer.
   - Encoder.
   - Scaler nếu cần.
- Folder:
   src/models/
   models/
   configs/model.yaml
   tests/test_model.py
============================================================
NGƯỜI 4 — INTEGRATION / PIPELINE / ARCHITECTURE
============================================================
- Branch: feature/pipeline
- VAI TRÒ: Data Science / Project Lead
- MỤC TIÊU: Kết nối toàn bộ phần việc của team thành một project hoàn chỉnh.
------------------------------------------------------------
1. INTEGRATION
------------------------------------------------------------
- Kết nối:
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
Model
 ↓
Evaluation
 ↓
Prediction
------------------------------------------------------------
2. PIPELINE
------------------------------------------------------------
- Xây dựng pipeline để các module hoạt động đúng thứ tự.
Raw Data
    ↓
Validation
    ↓
Cleaning
    ↓
Feature Engineering
    ↓
NLP
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
------------------------------------------------------------
3. CONFIGURATION
------------------------------------------------------------
configs/ là folder dùng chung.
Không có thành viên nào chỉ phụ trách Config.
Cấu trúc:
   configs/
   ├── data.yaml
   ├── features.yaml
   ├── model.yaml
   └── pipeline.yaml
data.yaml:
   - Data path.
   - Cleaning parameters.
   - Split parameters.
features.yaml:
   - Feature parameters.
   - NLP parameters.
   - TF-IDF parameters.
model.yaml:
   - Model parameters.
   - Hyperparameters.
pipeline.yaml:
   - Pipeline settings.
   - Output paths.
   - Experiment settings.
------------------------------------------------------------
4. UTILS
------------------------------------------------------------
src/utils/
Ví dụ:
   - config_loader.py
   - logger.py
   - random_seed.py
   - file_utils.py
Chứa các utility dùng chung.
------------------------------------------------------------
5. TESTING
------------------------------------------------------------
Kiểm tra:
   - Data module.
   - Feature module.
   - Model module.
   - Pipeline.
Cấu trúc:
   tests/
   ├── test_data.py
   ├── test_features.py
   ├── test_model.py
   └── test_pipeline.py
------------------------------------------------------------
6. GIT / TEAM INTEGRATION
------------------------------------------------------------
Quản lý:
   - Pull Request.
   - Merge.
   - Conflict.
   - Code review.
   - Kiểm tra code sau khi merge.
   - Đảm bảo develop luôn chạy được.
------------------------------------------------------------
7. DOCUMENTATION
------------------------------------------------------------
README.md phải mô tả:
   - Business Problem.
   - Dataset.
   - Data Understanding.
   - Data Cleaning.
   - EDA.
   - Statistical Analysis.
   - Relationship Analysis.
   - Feature Engineering.
   - NLP.
   - Modeling.
   - Evaluation.
   - Model Interpretation.
   - Business Insights.
============================================================
Ý NGHĨA CÁC FOLDER
============================================================
.vscode/
   → Cấu hình VS Code của project.
configs/
   → Tham số cấu hình của project.
   → Không chứa logic xử lý chính.
data/
   → Dữ liệu.
data/raw/
   → Dữ liệu gốc.
data/interim/
   → Dữ liệu trung gian.
data/processed/
   → Dữ liệu đã xử lý.
models/
   → Model và artifact đã train.
notebooks/
   → Khám phá, EDA, phân tích và thử nghiệm.
scripts/
   → Các script chạy tác vụ độc lập.
src/
   → Source code chính.
src/data/
   → Collection, Validation, Cleaning.
src/features/
   → Feature Engineering và NLP.
src/visualization/
   → Visualization và các hàm phân tích/EDA có thể tái sử dụng.
src/models/
   → Training, Evaluation, Prediction.
src/pipeline/
   → Điều phối các module thành workflow hoàn chỉnh.
src/utils/
   → Utility dùng chung.
tests/
   → Unit Test và Integration Test.
============================================================
NGUYÊN TẮC CHUNG
============================================================
1. Không commit trực tiếp vào main.
2. Mỗi thành viên làm việc trên branch riêng.
3. Các branch được phát triển từ develop.
4. Hoàn thành task:

feature/xxx
    ↓
Pull Request
    ↓
develop
    ↓
Test
    ↓
main

5. Notebook dùng cho:
   - Exploration.
   - EDA.
   - Experiment.
   - Visualization.
   - Phân tích.
Logic có thể tái sử dụng phải đưa vào src/.
6. Không hard-code các parameter quan trọng.
Các parameter có thể thay đổi nên đưa vào configs/.
7. Không được chỉ tập trung vào Model.
Project phải bao gồm:
Data Collection
+
Data Cleaning
+
EDA
+
Statistical Analysis
+
Relationship Analysis
+
Feature Engineering
+
NLP
+
Modeling
+
Evaluation
+
Model Interpretation
+
Business Insights
8. TF-IDF chỉ là một kỹ thuật NLP, không phải toàn bộ
Feature Engineering.
9. EDA phải trả lời được các câu hỏi về:
   - Distribution.
   - Outlier.
   - Correlation.
   - Linear Relationship.
   - Categorical Relationship.
   - Numerical-Categorical Relationship.
   - Multivariate Relationship.
   - Class Imbalance.
   - Salary.
   - Location.
   - Work Type.
   - Skills.
   - Text.
10. Modeling phải phân tích:
   - Bias.
   - Variance.
   - Underfitting.
   - Overfitting.
   - Class Imbalance.
   - Model Performance.
   - Feature Importance.
============================================================
MỤC TIÊU CUỐI CÙNG
============================================================
JOB POSTINGS
      ↓
DATA UNDERSTANDING
      ↓
DATA CLEANING
      ↓
EDA
      ↓
STATISTICAL / RELATIONSHIP ANALYSIS
      ↓
BUSINESS INSIGHTS
      ↓
FEATURE ENGINEERING
      ↓
NLP + SKILL ANALYSIS
      ↓
MACHINE LEARNING
      ↓
MODEL EVALUATION
      ↓
MODEL INTERPRETATION
      ↓
JOB LEVEL PREDICTION
      ↓
BUSINESS INSIGHTS