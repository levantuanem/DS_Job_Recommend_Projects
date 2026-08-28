============================================================
TOPIC: JOB LEVEL ANALYSIS AND PREDICTION BASED ON JOB POSTING CONTENT AND FEATURES
============================================================
PHÂN CHIA NHIỆM VỤ PROJECT DS_JOB_RECOMMEND_PROJECTS
============================================================
MỤC TIÊU PROJECT
---
Phân tích dữ liệu tuyển dụng từ LinkedIn Job Postings để:
1. Hiểu đặc điểm của dữ liệu và thị trường tuyển dụng.
2. Phân tích các yếu tố liên quan đến Job Level.
3. Phân tích mối quan hệ giữa Feature với Feature.
4. Phân tích mối quan hệ giữa Feature với Target.
5. Phân tích các kỹ năng và yêu cầu trong từng Job Level.
6. Xây dựng Feature Engineering từ dữ liệu numerical, categorical và text.
7. Sử dụng NLP để khai thác thông tin từ nội dung Job Posting.
8. Xây dựng Machine Learning Model để dự đoán Job Level.
9. Phân tích Loss Function, Bias và Variance.
10. Phát hiện Underfitting và Overfitting.
11. Đánh giá và so sánh các Machine Learning Model.
12. Giải thích Model và xác định các Feature quan trọng.
13. Xây dựng Pipeline có thể chạy lại từ đầu đến cuối.
14. Đưa ra Business Insights từ dữ liệu và Model.
============================================================
DATA SCIENCE WORKFLOW
============================================================
Business Understanding
```
    ↓
```
Data Collection
```
    ↓
```
Data Understanding
```
    ↓
```
Data Quality Analysis
```
    ↓
```
Data Cleaning
```
    ↓
```
EDA
```
    ↓
```
Feature ↔ Feature Analysis
```
    ↓
```
Feature ↔ Target Analysis
```
    ↓
```
Statistical Analysis
```
    ↓
```
Feature Engineering
```
    ↓
```
NLP / Text Processing
```
    ↓
```
Skill Extraction
```
    ↓
```
Feature Selection
```
    ↓
```
Preprocessing
```
    ↓
```
Modeling
```
    ↓
```
Loss Function
```
    ↓
```
Cross Validation
```
    ↓
```
Bias / Variance Analysis
```
    ↓
```
Hyperparameter Tuning
```
    ↓
```
Evaluation
```
    ↓
```
Model Interpretation
```
    ↓
```
Prediction
```
    ↓
```
Business Insights
============================================================
NGƯỜI 1 — DATA COLLECTION + DATA UNDERSTANDING + DATA CLEANING
============================================================
* Branch: feature/data
* VAI TRÒ: Data Engineer / Data Analyst
* MỤC TIÊU: Thu thập, kiểm tra, hiểu và làm sạch dữ liệu tuyển dụng để tạo ra dataset có chất lượng, sẵn sàng cho EDA, Feature Engineering và Modeling.
---
1. DATA COLLECTION
---
* Kiểm tra nguồn dataset.
* Đưa dataset gốc vào: data/raw/
* Không chỉnh sửa trực tiếp dữ liệu raw.
* Ghi nhận:
  * Nguồn dữ liệu.
  * Tên dataset.
  * Số lượng record.
  * Số lượng feature.
  * Thời gian / phạm vi dữ liệu nếu có.
---
2. DATA UNDERSTANDING
---
* Kiểm tra:
  * Shape.
  * Số dòng / số cột.
  * Column names.
  * Data types.
  * Missing values.
  * Unique values.
  * Cardinality.
  * Các giá trị bất thường.
* Phân loại feature:
  * JOB INFORMATION:
    * title
    * description
    * skills_desc
    * formatted_experience_level
  * SALARY:
    * min_salary
    * med_salary
    * max_salary
    * normalized_salary
    * currency
    * pay_period
    * compensation_type
  * LOCATION:
    * location
    * zip_code
    * fips
  * WORK:
    * formatted_work_type
    * work_type
    * remote_allowed
  * ENGAGEMENT:
    * views
    * applies
  * TIME:
    * listed_time
    * original_listed_time
    * expiry
    * closed_time
  * COMPANY:
    * company information
---
3. DATA QUALITY ANALYSIS
---
* Kiểm tra:
  * Missing values.
  * Duplicate.
  * Sai datatype.
  * Giá trị không hợp lệ.
  * Category không đồng nhất.
  * Text bị lỗi.
  * Giá trị âm / bằng 0 không hợp lý.
  * Giá trị bất thường.
* Phải ghi rõ:
  * Vấn đề là gì?
  * Có bao nhiêu record bị ảnh hưởng?
  * Feature nào bị ảnh hưởng?
  * Mức độ ảnh hưởng?
  * Cách xử lý?
  * Tại sao chọn cách xử lý đó?
---
4. DATA CLEANING
---
* Xử lý:
  * Missing values.
  * Duplicate.
  * Sai datatype.
  * Giá trị không hợp lệ.
  * Category không đồng nhất.
  * Text bị lỗi.
  * Giá trị không hợp lý.
* Phân biệt:
  * Data Error.
  * Genuine Extreme Value.
* Không tự động xóa dữ liệu chỉ vì đó là outlier.
* Output: data/processed/
---
5. DATA DOCUMENTATION
---
* Ghi lại:
  * Dataset ban đầu.
  * Data Quality Issues.
  * Cleaning methods.
  * Số record trước / sau cleaning.
  * Column được giữ.
  * Column bị loại bỏ.
  * Lý do xử lý.
---
6. OUTPUT
---
* Clean Dataset.
* Data Quality Report.
* Data Understanding Notebook.
* Data Cleaning Notebook.
* Cleaning Functions.
* Dataset Documentation.
---
FOLDER LIÊN QUAN
---
data/raw/
data/processed/
notebooks/
src/data/
tests/test_data.py
============================================================
NGƯỜI 2 — EDA + STATISTICAL ANALYSIS + FEATURE RELATIONSHIP + VISUALIZATION
============================================================
* Branch: feature/visualization
* VAI TRÒ: Data Analyst / Visualization Analyst
* MỤC TIÊU: Phân tích dữ liệu đã clean, tìm ra các mối quan hệ giữa Feature với Feature và Feature với Target, trực quan hóa dữ liệu và đưa ra Business Insights.
---
1. UNIVARIATE ANALYSIS
---
Phân tích từng Feature riêng lẻ.
* NUMERICAL:
  * Salary.
  * Views.
  * Applies.
  * Normalized Salary.
  * Các numerical feature khác.
* Phân tích:
  * Mean.
  * Median.
  * Standard Deviation.
  * Min / Max.
  * Quartile.
  * IQR.
  * Distribution.
  * Skewness.
  * Outlier.
* Visualization:
  * Histogram.
  * KDE.
  * Boxplot.
* CATEGORICAL:
  * Job Level.
  * Work Type.
  * Employment Type.
  * Location.
  * Remote.
  * Pay Period.
* Phân tích:
  * Frequency.
  * Percentage.
  * Cardinality.
  * Category imbalance.
* Visualization:
  * Bar Chart.
  * Count Plot.
  * Percentage Chart.
---
2. FEATURE ↔ FEATURE ANALYSIS
---
Phân tích mối quan hệ giữa các Feature.
* NUMERICAL ↔ NUMERICAL:
  * Salary ↔ Views.
  * Salary ↔ Applies.
  * Views ↔ Applies.
  * Salary ↔ các numerical features.
  * Views ↔ các numerical features.
  * Applies ↔ các numerical features.
* Sử dụng:
  * Pearson Correlation.
  * Spearman Correlation.
* Xác định:
  * Positive Correlation.
  * Negative Correlation.
  * Weak Correlation.
  * Moderate Correlation.
  * Strong Correlation.
* Kiểm tra Multicollinearity.
* Xác định các Feature có thông tin trùng lặp hoặc tương quan quá cao.
* Visualization:
  * Correlation Matrix.
  * Correlation Heatmap.
  * Scatter Plot.
  * Pair Plot nếu phù hợp.
---
3. FEATURE ↔ TARGET ANALYSIS
---
Target: formatted_experience_level
Phân tích mối quan hệ giữa Feature và Job Level.
* NUMERICAL FEATURE ↔ TARGET:
  * Salary ↔ Job Level.
  * Views ↔ Job Level.
  * Applies ↔ Job Level.
  * Normalized Salary ↔ Job Level.
  * Các numerical feature khác ↔ Job Level.
* Phân tích:
  * Mean theo từng Job Level.
  * Median theo từng Job Level.
  * Distribution theo từng Job Level.
  * Difference giữa các nhóm.
* CATEGORICAL FEATURE ↔ TARGET:
  * Work Type ↔ Job Level.
  * Remote ↔ Job Level.
  * Location ↔ Job Level.
  * Employment Type ↔ Job Level.
* TEXT FEATURE ↔ TARGET:
  * Description Length ↔ Job Level.
  * Title Length ↔ Job Level.
  * Skill Count ↔ Job Level.
  * Keyword Frequency ↔ Job Level.
* Có thể sử dụng:
  * ANOVA.
  * Kruskal-Wallis.
  * Chi-square.
  * Effect Size.
  * Group Comparison.
* Mục tiêu:
  * Xác định Feature nào có mối quan hệ với Target.
  * Xác định mức độ khác biệt giữa các Job Level.
  * Xác định Feature nào có khả năng hữu ích cho Modeling.
* Lưu ý:
  Correlation / Association không đồng nghĩa với Causal Relationship.
---
4. CATEGORICAL ↔ NUMERICAL ANALYSIS
---
* Phân tích:
  * Job Level ↔ Salary.
  * Work Type ↔ Salary.
  * Remote ↔ Salary.
  * Location ↔ Salary.
  * Job Level ↔ Views.
  * Job Level ↔ Applies.
* Visualization:
  * Boxplot.
  * Violin Plot.
  * Bar Chart.
  * Grouped Bar Chart.
---
5. CATEGORICAL ↔ CATEGORICAL ANALYSIS
---
* Phân tích:
  * Job Level ↔ Work Type.
  * Job Level ↔ Remote.
  * Job Level ↔ Employment Type.
  * Job Level ↔ Location.
* Sử dụng:
  * Crosstab.
  * Percentage.
  * Chi-square Test nếu phù hợp.
* Visualization:
  * Stacked Bar Chart.
  * Grouped Bar Chart.
---
6. MULTIVARIATE ANALYSIS
---
* Phân tích nhiều Feature cùng lúc.
* Phân tích:
  * Interaction giữa các Feature.
  * Correlation giữa nhiều Numerical Feature.
  * Multicollinearity.
  * Relationship giữa nhiều Feature với Target.
  * Sự khác biệt giữa nhiều nhóm Job Level.
* Visualization:
  * Correlation Heatmap.
  * Pair Plot nếu phù hợp.
  * Grouped Visualization.
---
7. OUTLIER ANALYSIS
---
* Phân tích Outlier trong:
  * Salary.
  * Views.
  * Applies.
  * Các Numerical Feature khác.
* Xác định:
  * Outlier là lỗi dữ liệu hay giá trị thực tế?
  * Có nên Remove không?
  * Có nên Transform không?
* Có thể sử dụng:
  * IQR.
  * Z-score.
  * Log Transformation.
---
8. CLASS IMBALANCE ANALYSIS
---
Target:
formatted_experience_level
* Kiểm tra:
  * Entry Level.
  * Associate.
  * Mid-Senior Level.
  * Director.
  * Executive.
  * Internship.
* Phân tích:
  * Số lượng từng Class.
  * Tỷ lệ từng Class.
  * Mức độ Class Imbalance.
* Visualization:
  * Class Distribution.
  * Percentage Chart.
* Kết quả được bàn giao cho Người 4 để xử lý trong Modeling.
---
9. SKILL ANALYSIS
---
* Phân tích:
  * Skill phổ biến nhất.
  * Skill theo Job Level.
  * Skill theo Location.
  * Skill theo Work Type.
  * Skill nào xuất hiện nhiều ở Senior?
  * Skill nào xuất hiện nhiều ở Director?
  * Skill nào đặc trưng cho Entry Level?
* Nguồn:
  * skills_desc.
  * description.
  * title.
* Người 2 chỉ thực hiện Skill Analysis.
* Người 3 thực hiện Skill Extraction để tạo Feature cho Machine Learning.
---
10. TEXT ANALYSIS
---
* Phân tích:
  * Title Length.
  * Description Length.
  * Word Count.
  * Skill Count.
  * Text Length theo Job Level.
  * Keyword Frequency.
  * Keyword theo Job Level.
* Mục tiêu:
  * Hiểu đặc điểm nội dung Job Posting.
  * Tìm sự khác biệt về Text giữa các Job Level.
---
11. BUSINESS INSIGHTS
---
* Trả lời:
  * Job Level nào tuyển nhiều nhất?
  * Job Level nào có Salary cao nhất?
  * Salary có quan hệ với Job Level không?
  * Feature nào tương quan mạnh với Feature khác?
  * Có Multicollinearity không?
  * Feature nào có quan hệ với Target?
  * Skill nào phổ biến nhất?
  * Skill nào đặc trưng cho từng Job Level?
  * Location nào tuyển nhiều?
  * Location nào có Salary cao?
  * Remote Job tập trung ở Job Level nào?
---
12. OUTPUT
---
* EDA Notebooks.
* Statistical Analysis.
* Feature ↔ Feature Analysis.
* Feature ↔ Target Analysis.
* Correlation Analysis.
* Multicollinearity Analysis.
* Skill Analysis.
* Text Analysis.
* Visualization.
* Business Insights.
---
FOLDER LIÊN QUAN
---
notebooks/
src/visualization/
tests/
============================================================
NGƯỜI 3 — FEATURE ENGINEERING + NLP + SKILL EXTRACTION
============================================================
* Branch: feature/features
* VAI TRÒ: Feature Engineer / NLP Engineer
* MỤC TIÊU: Chuyển dữ liệu đã clean thành các Feature có thể sử dụng cho Machine Learning.
---
1. NUMERICAL FEATURE ENGINEERING
---
* Xử lý:
  * min_salary.
  * med_salary.
  * max_salary.
  * normalized_salary.
  * views.
  * applies.
* Có thể tạo:
  * salary_range.
  * log_salary.
  * application_rate.
  * view_to_apply_ratio.
* Phải kiểm tra Data Leakage trước khi sử dụng Views / Applies.
---
2. CATEGORICAL FEATURE ENGINEERING
---
* Xử lý:
  * work_type.
  * formatted_work_type.
  * remote_allowed.
  * pay_period.
  * currency.
  * location.
  * application_type.
* Có thể sử dụng:
  * One-Hot Encoding.
  * Frequency Encoding.
  * Ordinal Encoding nếu phù hợp.
---
3. TEXT FEATURE ENGINEERING
---
* Nguồn:
  * title.
  * description.
  * skills_desc.
* Tạo:
  * text_length.
  * word_count.
  * sentence_count.
  * skill_count.
  * keyword_count.
  * technical_keyword_count.
  * management_keyword_count.
  * leadership_keyword_count.
---
4. NLP
---
* Thực hiện:
  * Text Cleaning.
  * Lowercase.
  * Remove Noise.
  * Tokenization.
  * Stopword Removal nếu phù hợp.
  * N-gram.
  * TF-IDF.
* Có thể thử thêm phương pháp NLP khác nếu phù hợp.
* LƯU Ý:
  TF-IDF chỉ là một kỹ thuật NLP.
  NLP không đồng nghĩa với TF-IDF.
---
5. SKILL EXTRACTION
---
* Xác định và trích xuất Skill từ:
  * title.
  * description.
  * skills_desc.
* Có thể xây dựng:
  has_python
  has_sql
  has_java
  has_aws
  has_excel
  has_machine_learning
  has_cloud
  has_leadership
  has_management
  ...
* Skill Extraction phải tạo ra Feature có thể đưa vào Model.
---
6. FEATURE SELECTION
---
* Dựa trên kết quả EDA và Statistical Analysis của Người 2.
* Kiểm tra:
  * Feature hữu ích.
  * Feature dư thừa.
  * Feature tương quan quá cao với nhau.
  * Multicollinearity.
  * Feature gây Data Leakage.
* Có thể sử dụng:
  * Correlation.
  * Variance.
  * Mutual Information.
  * SelectKBest.
  * Feature Importance.
---
7. DATA LEAKAGE PREVENTION
---
* Kiểm tra:
  * Target Leakage.
  * Train/Test Leakage.
  * Information Leakage.
  * Feature được tạo từ thông tin chỉ có sau thời điểm dự đoán.
* Đặc biệt kiểm tra:
  * views.
  * applies.
  * các biến phát sinh sau khi Job Posting được đăng.
---
8. PREPROCESSING
---
* Xử lý:
  * Scaling.
  * Encoding.
  * Imputation nếu cần.
  * Text Vectorization.
* Xây dựng Preprocessing Pipeline.
* Đảm bảo:
  Training Data và Test Data
  sử dụng cùng preprocessing logic nhưng chỉ Fit trên Training Data.
---
9. OUTPUT
---
* Numerical Features.
* Categorical Features.
* Text Features.
* Skill Features.
* NLP Features.
* Feature Selection Results.
* Preprocessing Pipeline.
* X.
* y.
* Feature Specification.
---
FOLDER LIÊN QUAN
---
src/features/
configs/features.yaml
tests/test_features.py
notebooks/
============================================================
NGƯỜI 4 — MODELING + LOSS FUNCTION + BIAS / VARIANCE + EVALUATION
============================================================
* Branch: feature/model
* VAI TRÒ: Machine Learning Engineer
* MỤC TIÊU: Xây dựng, tối ưu, đánh giá và giải thích Machine Learning Model dùng để dự đoán Job Level.
---
1. TARGET
---
Target:
formatted_experience_level
Các Class:
```
- Entry Level.
- Associate.
- Mid-Senior Level.
- Director.
- Executive.
- Internship.
```
---
2. BASELINE MODEL
---
* Xây dựng Baseline Model.
* Có thể sử dụng:
  * Logistic Regression.
* Mục đích:
  * Tạo baseline performance.
  * Làm mốc so sánh với các Model phức tạp hơn.
---
3. MACHINE LEARNING MODELS
---
* Thử nghiệm:
  * Logistic Regression.
  * Random Forest.
  * Linear SVM.
  * Gradient Boosting.
  * XGBoost / LightGBM nếu phù hợp.
* So sánh các Model trên cùng Evaluation Strategy.
---
4. LOSS FUNCTION
---
* Phân tích Loss Function phù hợp với bài toán Multiclass Classification.
* Xem xét:
  * Cross-Entropy Loss / Log Loss.
  * Weighted Cross-Entropy nếu sử dụng Class Weight.
* Phân tích:
  * Loss dùng để tối ưu Model.
  * Ý nghĩa của Loss.
  * Training Loss.
  * Validation Loss nếu Model hỗ trợ.
  * Sự thay đổi của Loss trong quá trình Training
* Sử dụng Loss để hỗ trợ phân tích:
  * Underfitting.
  * Overfitting.
---
5. TRAINING
---
* Train/Test Split.
* Cross Validation.
* Training.
* Model Comparison.
* Đảm bảo Preprocessing chỉ được Fit trên Training Data.
* Tránh Data Leakage.
---
6. CLASS IMBALANCE
---
* Dựa trên kết quả Class Imbalance Analysis của Người 2.
* Có thể thử:
  * class_weight.
  * Random Over Sampling.
  * SMOTE nếu phù hợp.
* Đảm bảo Sampling chỉ được thực hiện trên Training Data.
---
7. CROSS VALIDATION
---
* Sử dụng Cross Validation để:
  * Đánh giá độ ổn định của Model.
  * So sánh các Model.
  * Phát hiện Model có Variance cao.
  * Hỗ trợ Hyperparameter Tuning.
* Theo dõi:
  * Mean score.
  * Standard deviation.
  * Score giữa các Fold.
---
8. HYPERPARAMETER TUNING
---
* Sử dụng:
  * GridSearchCV.
  * RandomizedSearchCV.
* Tối ưu:
  * Regularization.
  * Tree Depth.
  * Number of Estimators.
  * Learning Rate.
  * Các Hyperparameter phù hợp với Model.
---
9. BIAS ANALYSIS
---
Phân tích High Bias / Low Bias.
* So sánh:
  * Training Performance.
  * Validation Performance.
  * Test Performance.
* Dấu hiệu High Bias:
  * Training Performance thấp.
  * Validation Performance thấp.
  * Test Performance thấp.
* Kết luận:
  High Bias → Underfitting.
---
10. VARIANCE ANALYSIS
---
Phân tích High Variance / Low Variance.
* So sánh:
  * Training Performance.
  * Validation Performance.
  * Test Performance.
  * Cross Validation Scores.
  * Training Loss.
  * Validation Loss nếu có.
* Dấu hiệu High Variance:
  * Training Performance rất cao.
  * Validation / Test Performance thấp.
  * Training Loss thấp nhưng Validation Loss cao.
* Kết luận:
  High Variance → Overfitting.
---
11. UNDERFITTING / OVERFITTING
---
* Phân tích:
  * Underfitting.
  * Good Fit.
  * Overfitting.
* Dựa trên:
  * Training Score.
  * Validation Score.
  * Test Score.
  * Training Loss.
  * Validation Loss.
  * Cross Validation.
* Đề xuất hướng xử lý:
  Underfitting:
  ```
    - Model phức tạp hơn.
    - Feature tốt hơn.
    - Giảm Regularization nếu phù hợp.
  ```
  Overfitting:
  ```
    - Regularization.
    - Giảm Model Complexity.
    - Feature Selection.
    - Cross Validation.
    - More Data nếu có thể.
  ```
---
12. EVALUATION
---
* Không chỉ sử dụng Accuracy.
* Đánh giá:
  * Accuracy.
  * Precision.
  * Recall.
  * F1-score.
  * Macro-F1.
  * Weighted-F1.
  * Confusion Matrix.
* Với Class Imbalance:
  Ưu tiên Macro-F1.
---
13. MODEL COMPARISON
---
Tạo bảng:
```
Model
Accuracy
Precision
Recall
Macro-F1
Weighted-F1
Training Time
CV Mean
CV Std
```
* Chọn Best Model dựa trên Metric phù hợp với Business Problem.
---
14. MODEL INTERPRETATION
---
Trả lời:
"Model dựa vào những Feature nào để xác định Job Level?"
* Có thể sử dụng:
  * Feature Importance.
  * Permutation Importance.
  * SHAP.
  * Coefficient Analysis.
* Phân tích:
  * Numerical Feature quan trọng.
  * Categorical Feature quan trọng.
  * Skill quan trọng.
  * Keyword quan trọng.
  * Text Feature quan trọng.
---
15. MODEL ARTIFACT
---
* Lưu:
  * Best Model.
  * Vectorizer.
  * Encoder.
  * Scaler nếu cần.
  * Label Mapping.
* Đảm bảo các Artifact có thể sử dụng lại cho Prediction.
---
16. OUTPUT
---
* Baseline Results.
* Model Comparison.
* Best Model.
* Loss Analysis.
* Bias Analysis.
* Variance Analysis.
* Underfitting / Overfitting Analysis.
* Evaluation Report.
* Confusion Matrix.
* Hyperparameter Results.
* Feature Importance.
* Model Interpretation.
* Model Artifacts.
---
FOLDER LIÊN QUAN
---
src/models/
models/
configs/model.yaml
tests/test_model.py
notebooks/
============================================================
NGƯỜI 5 — INTEGRATION + PIPELINE + TESTING + DOCUMENTATION
============================================================

* Branch: feature/pipeline

* VAI TRÒ: Data Science / ML Engineer / Project Integration

* MỤC TIÊU: Kết nối toàn bộ phần việc của Team thành một Project hoàn chỉnh, có cấu trúc rõ ràng, có thể chạy lại và kiểm thử.

---

1. PROJECT ARCHITECTURE

---

Thiết kế và duy trì cấu trúc:

```
src/

├── data/

├── features/

├── visualization/

├── models/

├── pipeline/

└── utils/
```

---

2. INTEGRATION

---

Kết nối:

```
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

* Đảm bảo module của Người 1, Người 2, Người 3 và Người 4 hoạt động thống nhất.

---

3. PIPELINE

---

Xây dựng Pipeline:

```
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

4. CONFIGURATION

---

configs/

```
├── data.yaml

├── features.yaml

├── model.yaml

└── pipeline.yaml
```

* data.yaml:

  * Data Path.

  * Cleaning Parameters.

  * Split Parameters.

* features.yaml:

  * Feature Parameters.

  * NLP Parameters.

  * TF-IDF Parameters.

* model.yaml:

  * Model Parameters.

  * Hyperparameters.

  * Evaluation Parameters.

* pipeline.yaml:

  * Pipeline Settings.

  * Output Paths.

  * Experiment Settings.

* Người 5 chịu trách nhiệm tích hợp Config.

* Các thành viên khác chịu trách nhiệm cung cấp Parameter liên quan đến module của mình.

---

5. UTILS

---

src/utils/

```
- config_loader.py

- logger.py

- random_seed.py

- file_utils.py
```

* Chứa Utility dùng chung cho toàn Project.

---

6. TESTING

---

Kiểm tra:

```
- Data Module.

- Feature Module.

- Visualization Module nếu cần.

- Model Module.

- Pipeline.
```

* Cấu trúc:

  tests/

  ├── test_data.py

  ├── test_features.py

  ├── test_model.py

  └── test_pipeline.py

* Mỗi thành viên chịu trách nhiệm viết Unit Test cho phần code của mình.

* Người 5 chịu trách nhiệm Integration Test.

---

7. GIT / TEAM INTEGRATION

---

* Không commit trực tiếp vào main.

* Mỗi thành viên làm việc trên Branch riêng.

* Branch được phát triển từ develop.

* Quy trình:

  feature/xxx

  ```
    ↓
  ```

  Pull Request

  ```
    ↓
  ```

  develop

  ```
    ↓
  ```

  Test

  ```
    ↓
  ```

  main

* Kiểm tra:

  * Merge Conflict.

  * Dependency.

  * Code Quality.

  * Project chạy được sau Merge.

---

8. EXPERIMENT MANAGEMENT

---

* Quản lý:

  * Dataset Version.

  * Feature Version.

  * Model Version.

  * Configuration.

  * Evaluation Results.

* Đảm bảo có thể biết:

  * Model nào.

  * Feature nào.

  * Config nào.

  * Dataset nào.

  tạo ra kết quả nào.

---

9. PREDICTION

---

Tạo:

```
scripts/

├── train.py

└── predict.py
```

* Input:

  Job Posting mới.

* Pipeline:

  Text Preprocessing

  ```
    ↓
  ```

  Feature Engineering

  ```
    ↓
  ```

  NLP

  ```
    ↓
  ```

  Preprocessing

  ```
    ↓
  ```

  Model

  ```
    ↓
  ```

  Prediction

* Output:

  Predicted Job Level

  *

  Probability nếu Model hỗ trợ.

---

10. DOCUMENTATION

---

README.md phải mô tả:

```
- Business Problem.

- Dataset.

- Data Understanding.

- Data Cleaning.

- EDA.

- Feature ↔ Feature Analysis.

- Feature ↔ Target Analysis.

- Statistical Analysis.

- Feature Engineering.

- NLP.

- Skill Extraction.

- Modeling.

- Loss Function.

- Bias.

- Variance.

- Underfitting.

- Overfitting.

- Evaluation.

- Model Interpretation.

- Business Insights.

- How to Run Project.

- How to Train Model.

- How to Make Prediction.
```

---

11. FINAL INTEGRATION TEST

---

Kiểm tra Project từ đầu đến cuối:

```
Raw Dataset

    ↓

Data Cleaning

    ↓

EDA

    ↓

Feature ↔ Feature Analysis

    ↓

Feature ↔ Target Analysis

    ↓

Feature Engineering

    ↓

NLP

    ↓

Training

    ↓

Loss Analysis

    ↓

Evaluation

    ↓

Bias / Variance Analysis

    ↓

Model Interpretation

    ↓

Prediction
```

* Đảm bảo Project có thể chạy lại từ đầu.

* Kiểm tra:

  * Dependency.

  * File Path.

  * Config.

  * Model Artifact.

  * Preprocessing.

  * Prediction.

  * Test.

============================================================

Ý NGHĨA CÁC FOLDER

============================================================

.vscode/

```
→ Cấu hình VS Code của Project.
```

configs/

```
→ Configuration của Project.
```

data/

```
→ Dữ liệu.
```

data/raw/

```
→ Dữ liệu gốc.
```

data/interim/

```
→ Dữ liệu trung gian.
```

data/processed/

```
→ Dữ liệu đã xử lý.
```

models/

```
→ Model và Model Artifacts.
```

notebooks/

```
→ Exploration, EDA, Analysis và Experiment.
```

scripts/

```
→ Script chạy các tác vụ độc lập.
```

src/

```
→ Source Code chính.
```

src/data/

```
→ Data Collection, Validation, Cleaning.
```

src/features/

```
→ Feature Engineering và NLP.
```

src/visualization/

```
→ Visualization và EDA functions.
```

src/models/

```
→ Training, Evaluation, Prediction, Interpretation.
```

src/pipeline/

```
→ Điều phối các module thành Workflow hoàn chỉnh.
```

src/utils/

```
→ Utility dùng chung.
```

tests/

```
→ Unit Test và Integration Test.
```

============================================================

NGUYÊN TẮC PHÂN CHIA TRÁCH NHIỆM

============================================================

1. Người 1 chịu trách nhiệm:

   Data Collection

   *

   Data Understanding

   *

   Data Quality

   *

   Data Cleaning

---

2. Người 2 chịu trách nhiệm:

   EDA

   *

   Statistical Analysis

   *

   Feature ↔ Feature

   *

   Feature ↔ Target

   *

   Correlation

   *

   Multicollinearity

   *

   Skill Analysis

   *

   Text Analysis

   *

   Visualization

   *

   Business Insights

---

3. Người 3 chịu trách nhiệm:

   Feature Engineering

   *

   NLP

   *

   Skill Extraction

   *

   Feature Selection

   *

   Preprocessing

   *

   Data Leakage Prevention

---

4. Người 4 chịu trách nhiệm:

   Modeling

   *

   Loss Function

   *

   Class Imbalance

   *

   Cross Validation

   *

   Hyperparameter Tuning

   *

   Bias

   *

   Variance

   *

   Underfitting

   *

   Overfitting

   *

   Evaluation

   *

   Feature Importance

   *

   Model Interpretation

---

5. Người 5 chịu trách nhiệm:

   Integration

   *

   Pipeline

   *

   Configuration

   *

   Utils

   *

   Testing

   *

   Experiment Management

   *

   Prediction

   *

   Documentation

   *

   Git Integration

============================================================

NGUYÊN TẮC QUAN TRỌNG

============================================================

1. Không commit trực tiếp vào main.

2. Mỗi thành viên làm việc trên Branch riêng.

3. Các Branch được phát triển từ develop.

4. Không tự ý thay đổi logic của module do thành viên khác phụ trách.

5. Notebook dùng cho Exploration, EDA và Experiment.

6. Logic có thể tái sử dụng phải đưa vào src/.

7. Không hard-code các Parameter quan trọng.

8. Parameter có thể thay đổi nên đưa vào configs/.

9. Không được chỉ tập trung vào Model.

10. Project phải bao gồm:

    Data Collection

    *

    Data Cleaning

    *

    EDA

    *

    Feature ↔ Feature

    *

    Feature ↔ Target

    *

    Statistical Analysis

    *

    Feature Engineering

    *

    NLP

    *

    Skill Extraction

    *

    Modeling

    *

    Loss Function

    *

    Bias / Variance

    *

    Evaluation

    *

    Model Interpretation

    *

    Prediction

    *

    Business Insights

11. Feature ↔ Feature dùng để hiểu:

    * Correlation.

    * Redundancy.

    * Multicollinearity.

12. Feature ↔ Target dùng để hiểu:

    * Feature nào có mối quan hệ với Job Level.

    * Mức độ khác biệt giữa các nhóm Target.

    * Feature nào có khả năng hữu ích cho Modeling.

13. Skill Analysis và Skill Extraction phải được phân biệt:

    Skill Analysis:

    ```
    Phân tích Skill để tìm Business Insight.
    ```

    Skill Extraction:

    ```
    Chuyển Skill thành Feature cho Machine Learning.
    ```

14. Text Analysis và NLP phải được phân biệt:

    Text Analysis:

    ```
    Phân tích đặc điểm Text.
    ```

    NLP:

    ```
    Xử lý và chuyển Text thành Feature cho Machine Learning.
    ```

15. Correlation không đồng nghĩa với Causation.

16. Không được đánh giá Model chỉ bằng Accuracy.

17. Khi Class Imbalance đáng kể, phải chú ý Macro-F1.

18. Phải phân tích Bias và Variance.

19. Phải kiểm tra Underfitting và Overfitting.

20. Phải kiểm tra Data Leakage trước khi Training.

============================================================

MỤC TIÊU CUỐI CÙNG

============================================================

JOB POSTINGS

```
  ↓
```

DATA UNDERSTANDING

```
  ↓
```

DATA CLEANING

```
  ↓
```

EDA

```
  ↓
```

FEATURE ↔ FEATURE

```
  ↓
```

FEATURE ↔ TARGET

```
  ↓
```

STATISTICAL ANALYSIS

```
  ↓
```
FEATURE ENGINEERING
```
  ↓
```
NLP + SKILL EXTRACTION
```
  ↓
```
FEATURE SELECTION
```
  ↓
```
PREPROCESSING
```
  ↓
```
MACHINE LEARNING
```
  ↓
```
LOSS FUNCTION
```
  ↓
```
BIAS / VARIANCE
```
  ↓
```
UNDERFITTING / OVERFITTING
```
  ↓
```
MODEL EVALUATION
```
  ↓
```
MODEL INTERPRETATION
```
  ↓
```
JOB LEVEL PREDICTION
```
  ↓
```
BUSINESS INSIGHTS
============================================================
PHÂN CHIA BRANCH CUỐI CÙNG
============================================================
develop

│

├── feature/data

│   → Người 1

│   → Data Collection + Data Understanding + Data Cleaning

│

├── feature/visualization

│   → Người 2

│   → EDA + Statistical Analysis + Feature Relationship + Visualization

│

├── feature/features

│   → Người 3

│   → Feature Engineering + NLP + Skill Extraction + Preprocessing

│

├── feature/model

│   → Người 4

│   → Modeling + Loss Function + Bias/Variance + Evaluation

│

└── feature/pipeline
```
→ Người 5
→ Integration + Pipeline + Testing + Documentation
```
============================================================
