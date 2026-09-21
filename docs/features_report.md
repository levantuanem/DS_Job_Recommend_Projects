# Feature Engineering & NLP

## 1. Overview

Feature Engineering module chịu trách nhiệm chuyển đổi dữ liệu job posting ban đầu thành tập đặc trưng có thể sử dụng trực tiếp cho Machine Learning.

Trong project `DS_Job_Recommend_Projects`, Feature Engineering tập trung vào bài toán dự đoán:

```text
formatted_experience_level
```

Các class mục tiêu gồm:

```text
Entry Level
Associate
Mid-Senior Level
Director
Executive
Internship
```

Dataset ban đầu có các trường như `title`, `description`, `skills_desc`, `listed_time`, `work_type`, `location`, `formatted_work_type`, `application_type`,...

Feature Engineering được tổ chức theo mô hình modular:

```text
                    Raw Job Data
                         │
                         ▼
                build_features.py
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
   text_features   skill_extraction   temporal_features
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                  Preprocessing
                         │
                         ▼
                  Feature Selection
                         │
                         ▼
                Machine Learning
```

Thiết kế này giúp mỗi module chỉ chịu trách nhiệm cho một nhóm feature riêng, trong khi `build_features.py` đóng vai trò điều phối toàn bộ workflow.

---

# 2. Module Structure

Các module chính:

```text
src/
└── features/
    ├── __init__.py
    ├── build_features.py
    ├── text_features.py
    ├── skill_extraction.py
    ├── temporal_features.py
    └── feature_selection.py
```

Vai trò của từng file:

| File                   | Responsibility                                        |
| ---------------------- | ----------------------------------------------------- |
| `build_features.py`    | Điều phối toàn bộ Feature Engineering pipeline        |
| `text_features.py`     | Tạo numerical features từ text và tạo `combined_text` |
| `skill_extraction.py`  | Trích xuất skill và tạo binary skill features         |
| `temporal_features.py` | Tạo feature từ `listed_time`                          |
| `feature_selection.py` | Chọn các feature có thông tin hữu ích                 |

Ngoài ra, kết quả Feature Engineering được đưa vào preprocessing pipeline để thực hiện imputation, scaling, encoding và TF-IDF trước khi đưa vào model. Documentation của project cũng xác định `src/features/`, `configs/features.yaml` và `tests/test_features.py` là các thành phần chính của Feature Engineering.

---

# 3. `build_features.py`

## 3.1. Purpose

`build_features.py` là module trung tâm của Feature Engineering.

Nó không trực tiếp thực hiện tất cả các phép biến đổi mà gọi các module chuyên biệt:

```text
build_features.py
       │
       ├── text_features.py
       ├── skill_extraction.py
       ├── temporal_features.py
       └── feature_selection.py
```

Mục tiêu của cách tổ chức này là tách:

```text
Feature implementation
```

khỏi:

```text
Pipeline orchestration
```

---

## 3.2. Main workflow

Workflow tổng quát:

```text
Load dataset
     ↓
Select target
     ↓
Remove rows with missing target
     ↓
Text Feature Engineering
     ↓
Skill Extraction
     ↓
Temporal Feature Engineering
     ↓
Drop unnecessary raw columns
     ↓
Train/Test Split
     ↓
Preprocessing
     ↓
Optional Feature Selection
     ↓
Save artifacts
```

---

## 3.3. Target handling

Target của bài toán là:

```python
formatted_experience_level
```

Các sample không có target hợp lệ cần được loại bỏ trước khi training.

Điều này đảm bảo model luôn có label để học supervised learning.

---

# 4. `text_features.py`

## 4.1. Purpose

`text_features.py` xử lý các trường văn bản:

```text
title
description
skills_desc
```

Các trường này chứa nhiều thông tin quan trọng về:

* Job title
* Job responsibilities
* Required skills
* Technical requirements
* Job description

Tuy nhiên, text raw không thể trực tiếp đưa vào các model Machine Learning thông thường.

Do đó module tạo thêm numerical features và một trường text tổng hợp.

---

## 4.2. Missing value handling

Trước khi xử lý text, các giá trị missing được chuyển thành chuỗi rỗng:

```python
df[col] = df[col].fillna("").astype(str)
```

Mục đích:

```text
NaN
 ↓
""
 ↓
Có thể sử dụng .str operations
```

Điều này tránh lỗi khi thực hiện các thao tác như:

```python
.str.len()
.str.split()
```

---

# 5. Text Length Features

## 5.1. Character length

Với mỗi text column, module tính độ dài chuỗi:

```text
title_length
description_length
skills_desc_length
```

Ví dụ:

```text
title = "Machine Learning Engineer"
```

thì:

```text
title_length = số ký tự của title
```

### Ý nghĩa

Độ dài text có thể cung cấp thông tin gián tiếp về loại job.

Ví dụ:

```text
Short title
→ thường chứa job role chính

Long description
→ có thể chứa nhiều requirements và responsibilities
```

---

# 6. Word Count

Module cũng tạo các feature dựa trên số lượng từ:

```text
title_word_count
description_word_count
skills_desc_word_count
```

Ví dụ:

```text
"Machine Learning Engineer"
```

có:

```text
3 words
```

Word count cung cấp thông tin khác với character length.

Ví dụ:

```text
"AI Engineer"
```

và:

```text
"Senior Machine Learning Engineer"
```

có thể có độ dài và số lượng từ khác nhau.

---

# 7. Combined Text

Một feature quan trọng là:

```text
combined_text
```

Các trường:

```text
title
description
skills_desc
```

được kết hợp thành một trường text chung.

Concept:

```text
title
   +
description
   +
skills_desc
   ↓
combined_text
```

Ví dụ:

```text
title:
Senior Machine Learning Engineer

description:
Develop machine learning systems...

skills_desc:
Python, PyTorch, SQL
```

được kết hợp thành:

```text
Senior Machine Learning Engineer
Develop machine learning systems...
Python, PyTorch, SQL
```

`combined_text` sau đó được sử dụng làm input cho TF-IDF.

---

# 8. NLP and TF-IDF

Cần phân biệt:

```text
NLP ≠ TF-IDF
```

TF-IDF chỉ là **một kỹ thuật vectorization trong NLP**.

Pipeline của project có thể được hiểu:

```text
Raw Text
   ↓
Text Cleaning / Normalization
   ↓
combined_text
   ↓
TF-IDF
   ↓
Numerical Vector
   ↓
Machine Learning Model
```

TF-IDF giúp biến text thành vector số để model có thể xử lý.

Ví dụ:

```text
"python machine learning"
```

có thể được chuyển thành:

```text
[0.12, 0.00, 0.37, ...]
```

Các giá trị phụ thuộc vào vocabulary và TF-IDF configuration.

---

# 9. `skill_extraction.py`

## 9.1. Purpose

`skill_extraction.py` tập trung vào việc xác định các technical/professional skills xuất hiện trong job posting.

Ví dụ:

```text
Python
SQL
Java
AWS
Machine Learning
Excel
Cloud
Leadership
Management
```

Skill extraction là một phần quan trọng vì job description thường chứa trực tiếp những requirements liên quan đến vị trí tuyển dụng.

---

# 10. Skill Binary Features

Skill extraction tạo các feature dạng binary.

Ví dụ:

```text
skill_python
skill_sql
skill_aws
skill_java
skill_machine_learning
```

Giá trị:

```text
1 → skill xuất hiện
0 → skill không xuất hiện
```

Ví dụ:

| combined_text             | skill_python | skill_sql | skill_aws |
| ------------------------- | -----------: | --------: | --------: |
| Python developer with SQL |            1 |         1 |         0 |
| AWS Cloud Engineer        |            0 |         0 |         1 |

Như vậy text không chỉ được biểu diễn bằng TF-IDF mà còn có các feature có ý nghĩa trực tiếp.

---

# 11. Skill Count

Ngoài từng skill riêng biệt, module tạo:

```text
skill_count
```

Feature này biểu diễn tổng số skill được phát hiện trong job posting.

Ví dụ:

```text
Python
SQL
AWS
Docker
Git
```

thì:

```text
skill_count = 5
```

Feature này giúp model nhận biết mức độ đa dạng của yêu cầu kỹ năng.

---

# 12. Why Skill Extraction and TF-IDF Are Both Used

Hai phương pháp có vai trò khác nhau.

### Skill extraction

Tạo feature có ý nghĩa rõ ràng:

```text
skill_python = 1
skill_sql = 1
skill_aws = 0
```

Ưu điểm:

* Dễ giải thích
* Dễ kiểm tra
* Có ý nghĩa domain rõ ràng

### TF-IDF

Cho phép model khai thác vocabulary rộng hơn:

```text
machine
learning
engineer
analytics
backend
cloud
...
```

Do đó pipeline sử dụng cả:

```text
Explicit skill features
+
TF-IDF text features
```

thay vì chỉ dựa vào một phương pháp.

---

# 13. `temporal_features.py`

## 13.1. Purpose

`temporal_features.py` xử lý:

```text
listed_time
```

Mục tiêu là chuyển timestamp thành những đặc trưng thời gian dễ sử dụng hơn.

Concept:

```text
listed_time
     ↓
Datetime
     ↓
┌───────────────┐
│ year          │
│ month         │
│ day           │
│ day_of_week   │
│ hour          │
│ weekend       │
└───────────────┘
```

---

# 14. Calendar Features

Từ `listed_time`, có thể tạo các feature theo calendar:

```text
listed_year
listed_month
listed_day
listed_dayofweek
```

Các feature này giúp model nhận biết temporal patterns trong dữ liệu job posting.

---

# 15. Weekend Feature

Một feature quan trọng:

```text
is_weekend
```

Logic:

```text
Saturday / Sunday
        ↓
is_weekend = 1

Monday - Friday
        ↓
is_weekend = 0
```

Feature này biến thông tin ngày trong tuần thành một binary feature dễ sử dụng.

---

# 16. `feature_selection.py`

## 16.1. Purpose

Sau khi tạo nhiều feature, không phải feature nào cũng cần thiết.

Feature Selection nhằm giảm:

```text
Redundant features
Irrelevant features
High-dimensional features
```

và giúp kiểm soát kích thước feature space.

Project định hướng sử dụng các phương pháp như:

```text
Correlation
Variance
Feature Importance
Mutual Information
SelectKBest
```

Trong implementation hiện tại, `SelectKBest` với `mutual_info_classif` được sử dụng cho bước selection khi được bật.

---

# 17. Mutual Information

Mutual Information đo mức độ phụ thuộc giữa feature và target.

Concept:

```text
Feature
   │
   ▼
Mutual Information
   │
   ▼
Relationship with Target
```

Feature có mutual information cao hơn có thể chứa nhiều thông tin hơn về target.

Trong classification:

```python
mutual_info_classif
```

được sử dụng để đánh giá feature.

---

# 18. SelectKBest

`SelectKBest` chọn ra `k` feature tốt nhất theo scoring function.

Concept:

```text
Original features
       │
       ▼
Mutual Information
       │
       ▼
Rank features
       │
       ▼
Select top K
       │
       ▼
Reduced feature set
```

Ví dụ:

```text
1000 features
     ↓
SelectKBest(k=200)
     ↓
200 features
```

Feature selection được thực hiện sau khi feature representation đã được xây dựng.

---

# 19. Preprocessing Pipeline

Sau Feature Engineering, các feature có nhiều kiểu dữ liệu khác nhau.

Ví dụ:

```text
Numerical
Categorical
Binary
Text
```

Do đó cần preprocessing riêng cho từng nhóm.

Concept:

```text
                    Features
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      Numerical    Categorical     Text
          │            │            │
          ▼            ▼            ▼
       Scaling       Encoding     TF-IDF
          │            │            │
          └────────────┼────────────┘
                       ▼
                Feature Matrix
```

Project documentation xác định preprocessing bao gồm:

* Missing value imputation
* Scaling
* Encoding
* Text vectorization

và sử dụng các thành phần như `Pipeline` và `ColumnTransformer`.

---

# 20. Numerical Features

Các numerical features được xử lý bằng:

```text
Imputation
    ↓
Scaling
```

Mục đích:

* Xử lý missing values
* Đưa numerical features về scale phù hợp

Ví dụ:

```text
title_length
description_length
skill_count
```

---

# 21. Categorical Features

Các categorical features có thể bao gồm:

```text
work_type
formatted_work_type
pay_period
currency
location
application_type
```

Các biến categorical được chuyển sang numerical representation bằng encoding phù hợp.

Một hướng được xác định trong project là:

```text
One-Hot Encoding
```

để model có thể sử dụng các category.

---

# 22. Binary Features

Các feature như:

```text
skill_python
skill_sql
skill_aws
is_weekend
```

đã có dạng:

```text
0 / 1
```

Do đó không cần chuyển đổi category theo cách giống categorical string features.

---

# 23. Text Features in Preprocessing

`combined_text` được đưa vào:

```text
TF-IDF Vectorizer
```

Pipeline:

```text
combined_text
      ↓
TF-IDF Vectorizer
      ↓
Sparse Matrix
```

Kết quả là một high-dimensional numerical representation.

Ví dụ:

```text
combined_text
     ↓
TF-IDF
     ↓
[
  0.00,
  0.31,
  0.00,
  0.12,
  ...
]
```

---

# 24. Data Leakage Prevention

Feature Engineering phải được thực hiện sao cho không sử dụng thông tin của test set để quyết định feature transformation.

Đặc biệt đối với:

```text
TF-IDF
Feature Selection
Scaling
Encoding
```

các parameters cần được fit trên training data.

Concept:

```text
Train Data
    │
    ├── fit preprocessing
    ├── fit TF-IDF
    └── fit feature selection
          │
          ▼
       Transform
          │
          ▼
       Test Data
```

Không nên:

```text
Train + Test
      ↓
fit TF-IDF
```

vì vocabulary và statistics của test data có thể ảnh hưởng vào quá trình training.

Tương tự, feature selection phải được fit trên training set thay vì toàn bộ dataset.

---

# 25. Feature Engineering Pipeline

Toàn bộ workflow có thể tóm tắt:

```text
                    RAW DATASET
                         │
                         ▼
              formatted_experience_level
                         │
                         ▼
                Remove missing target
                         │
                         ▼
              ┌──────────────────────┐
              │ Feature Engineering  │
              └──────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        TEXT           SKILLS         TIME
          │              │              │
          ▼              ▼              ▼
     text length    skill_python    year
     word count     skill_sql       month
     combined_text  skill_aws       day
          │          skill_count     weekday
          │              │           weekend
          └──────────────┼──────────────┘
                         ▼
                  Train / Test Split
                         │
                         ▼
                ┌─────────────────┐
                │ Preprocessing   │
                └─────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Numerical      Categorical       Text
       Scaling         Encoding        TF-IDF
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Feature Matrix
                         │
                         ▼
                 Feature Selection
                    (optional)
                         │
                         ▼
                  Machine Learning
```

---

# 26. Output Artifacts

Feature Engineering pipeline có thể tạo các artifact cần thiết cho inference.

Ví dụ:

```text
models/
├── preprocessor.pkl
└── feature_selector.pkl
```

`preprocessor.pkl` lưu preprocessing logic đã được fit.

`feature_selector.pkl` lưu feature-selection transformation nếu selection được bật.

Nhờ vậy, khi prediction trên dữ liệu mới, cùng preprocessing logic có thể được áp dụng:

```text
New Job
   ↓
Feature Engineering
   ↓
Saved Preprocessor
   ↓
Saved Feature Selector
   ↓
Model
   ↓
Prediction
```

---

# 27. Why the Architecture Is Modular

Thiết kế hiện tại có lợi ích:

### `text_features.py`

Chỉ tập trung vào:

```text
Text-derived features
```

### `skill_extraction.py`

Chỉ tập trung vào:

```text
Skill extraction
```

### `temporal_features.py`

Chỉ tập trung vào:

```text
Time-derived features
```

### `feature_selection.py`

Chỉ tập trung vào:

```text
Feature selection
```

### `build_features.py`

Điều phối:

```text
Complete feature pipeline
```

Điều này giúp code dễ:

* Maintain
* Test
* Debug
* Reuse
* Extend

và phù hợp với cách tổ chức project Machine Learning theo module.

---

# 28. Testing

Feature Engineering cần được kiểm thử tại:

```text
tests/test_features.py
```

Các nhóm test nên kiểm tra:

```text
Text features
    ↓
Skill features
    ↓
Temporal features
    ↓
Feature selection
    ↓
Complete feature pipeline
```

Ví dụ cần kiểm tra:

```text
Missing text
    → không gây error

Skill exists
    → skill feature = 1

Skill absent
    → skill feature = 0

Valid timestamp
    → temporal features được tạo

Invalid / missing timestamp
    → pipeline xử lý an toàn
```

---

# 29. Summary

Feature Engineering của project biến job posting raw data thành Machine Learning features thông qua bốn nhóm chính:

```text
1. Text Features
2. Skill Features
3. Temporal Features
4. Feature Selection
```

Trong đó:

```text
Text
 ↓
text_features.py
 ↓
combined_text + statistical text features
```

```text
Skills
 ↓
skill_extraction.py
 ↓
skill_* + skill_count
```

```text
Timestamp
 ↓
temporal_features.py
 ↓
calendar/time features
```

```text
Feature Matrix
 ↓
feature_selection.py
 ↓
Selected Features
```

Cuối cùng:

```text
Feature Engineering
        ↓
Preprocessing
        ↓
Feature Matrix
        ↓
Model Training
        ↓
Experience Level Prediction
```
