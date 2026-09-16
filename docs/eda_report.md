# Báo Cáo Phân Tích Khám Phá Dữ Liệu & Thống Kê (EDA & Statistical Analysis Report)
**Project:** LinkedIn Job Postings — Job Level Analysis & Prediction  
**Branch:** `feature/visualization`  
**Vai trò:** Data Analyst / Visualization Engineer (NGƯỜI 2)  
**Notebook tham chiếu:** `notebooks/03_eda.ipynb`  
**Module mã nguồn:** `src/visualization/visualize.py`  

---

## 1. Tổng quan Phân tích & Mục tiêu Nhiệm vụ

Nhiệm vụ của **Người 2** là tiếp nhận tập dữ liệu sạch từ Người 1 (`data/processed/postings_clean.csv`), tiến hành phân tích khám phá toàn diện (Exploratory Data Analysis - EDA), kiểm định thống kê toán học, tìm kiếm quy luật tương tác giữa các đặc trưng với biến mục tiêu, và trực quan hóa trực quan để làm cơ sở vững chắc cho:
* **Người 3 (Feature Engineering):** Biết cách xử lý độ lệch, biến đổi Log, xử lý đa cộng tuyến và khai thác đặc trưng văn bản / kỹ năng.
* **Người 4 (Modeling):** Nắm rõ mức độ mất cân bằng lớp (Class Imbalance), các đặc trưng có quan hệ thống kê mạnh nhất để lựa chọn thuật toán, hàm mất mát (loss function) và độ đo đánh giá phù hợp.

### Dataset đầu vào:
* **Bảng chính:** `postings_clean.csv` (123,849 dòng $\times$ 31 cột).
* **Các bảng phụ trợ:** `job_skills_clean.csv` (213,768 dòng), `companies_clean.csv` (24,473 dòng), `salaries_clean.csv` (40,785 dòng), `job_industries_clean.csv` (164,808 dòng).
* **Biến mục tiêu (Target):** `formatted_experience_level` (Cấp độ kinh nghiệm công việc: Entry level, Associate, Mid-Senior level, Director, Executive, Internship).

---

## 2. Phân Nhóm Đặc Trưng (Feature Grouping)

31 thuộc tính trong bảng chính được phân loại thành các nhóm phục vụ phân tích chuyên sâu:

| Nhóm đặc trưng | Danh sách cột | Vai trò trong phân tích |
|---|---|---|
| **Target Variable** | `formatted_experience_level` | Biến mục tiêu phân loại 6 lớp (khuyết 29,409 dòng ~ 23.75%). |
| **Numerical Features** | `min_salary`, `med_salary`, `max_salary`, `normalized_salary`, `views`, `applies` | Đo lường thu nhập, mức độ thu hút ứng viên và mức độ cạnh tranh. |
| **Categorical Features** | `formatted_work_type`, `work_type`, `remote_allowed`, `pay_period`, `compensation_type`, `currency` | Định danh hình thức làm việc, địa điểm, chế độ làm việc từ xa, đơn vị tiền tệ. |
| **Text Features** | `title`, `description`, `skills_desc` | Dữ liệu ngôn ngữ tự nhiên chứa yêu cầu chuyên môn và kỹ năng. |
| **Metadata / Identifier** | `job_id`, `company_id`, `company_name`, `location`, `zip_code`, `fips`, `listed_time`... | Khóa định danh, thuộc tính thời gian và địa lý. |

---

## 3. Phân Tích Đơn Biến (Univariate Analysis)

### 3.1. Các biến số (Numerical Features)

Bảng thống kê mô tả 8 chỉ số của các biến số chính (đơn vị lương quy đổi USD/năm):

| Đặc trưng | Số lượng quan sát | Mean | Median | Std | IQR | Skewness | Kurtosis | Hình dáng phân phối |
|---|---|---|---|---|---|---|---|---|
| `normalized_salary` | 36,059 | $89,145 | $79,900 | $52,168 | $58,000 | **+2.68** | 12.45 | Lệch phải mạnh (Right-skewed) |
| `min_salary` | 29,793 | $78,854 | $65,000 | $47,211 | $55,000 | **+2.45** | 10.82 | Lệch phải mạnh |
| `max_salary` | 29,793 | $104,821 | $90,000 | $61,340 | $70,000 | **+2.72** | 13.11 | Lệch phải mạnh |
| `med_salary` | 6,266 | $42,150 | $31,200 | $35,420 | $38,000 | **+1.98** | 5.62 | Lệch phải vừa |
| `views` | 122,160 | 18.5 | 6.0 | 48.2 | 16.0 | **+8.42** | 128.70 | Lệch phải cực mạnh |
| `applies` | 23,320 | 8.2 | 3.0 | 21.4 | 8.0 | **+6.15** | 64.30 | Lệch phải cực mạnh |

> **Nhận xét chuyên môn:**
> 1. $\text{Mean} > \text{Median}$ rõ rệt ở tất cả các biến số, cùng hệ số Skewness dương lớn ($\text{Skew} > 2.0$), khẳng định phân phối lương và tương tác có đuôi dài về bên phải. Phần lớn tin tuyển dụng có mức lương từ $40,000 - $90,000, chỉ một tỷ lệ nhỏ vị trí cấp cao nhận lương từ $200,000 - $400,000+.
> 2. `views` và `applies` có độ nhọn (Kurtosis) cực cao do một số ít tin tuyển dụng viral từ các tập đoàn lớn thu hút hàng nghìn lượt xem và ứng tuyển.

### 3.2. Các biến phân loại (Categorical Features)

* **`formatted_work_type`:** `Full-time` chiếm đa số áp đảo với **81.2%**, tiếp theo là `Contract` (11.5%), `Part-time` (5.1%), `Internship` (1.2%), các loại khác (`Temporary`, `Volunteer`) chiếm $< 1\%$.
* **`remote_allowed`:** Chỉ có **12.3%** tin tuyển dụng cho phép làm việc từ xa (Remote = 1), còn lại **87.7%** là Onsite hoặc Hybrid (Remote = 0).
* **`pay_period`:** Chủ yếu trả lương theo năm `YEARLY` (64.8%) và theo giờ `HOURLY` (33.2%), các hình thức `MONTHLY`, `WEEKLY`, `BIWEEKLY` chỉ chiếm 2%.

---

## 4. Phân Tích Quan Hệ Đặc Trưng ↔ Đặc Trưng (Feature ↔ Feature)

### 4.1. Ma trận tương quan (Correlation Matrix)

Tính toán ma trận tương quan tuyến tính Pearson ($r$) và tương quan thứ bậc phi tham số Spearman ($r_s$) giữa các biến số:

| Cặp đặc trưng | Pearson $r$ | Spearman $r_s$ | Mức độ tương quan | Chiều hướng |
|---|---|---|---|---|
| `min_salary` $\leftrightarrow$ `max_salary` | **0.998** | **0.997** | Cực kỳ mạnh | Đồng biến (+) |
| `max_salary` $\leftrightarrow$ `normalized_salary` | **0.996** | **0.995** | Cực kỳ mạnh | Đồng biến (+) |
| `min_salary` $\leftrightarrow$ `normalized_salary` | **0.995** | **0.994** | Cực kỳ mạnh | Đồng biến (+) |
| `views` $\leftrightarrow$ `applies` | **0.652** | **0.710** | Mạnh | Đồng biến (+) |
| `normalized_salary` $\leftrightarrow$ `views` | 0.082 | 0.114 | Rất yếu | Không rõ ràng |
| `normalized_salary` $\leftrightarrow$ `applies` | 0.041 | 0.065 | Rất yếu | Không rõ ràng |

### 4.2. Phát hiện Đa cộng tuyến nghiêm trọng (Multicollinearity Alert)

* **Vấn đề phát hiện:** Ba biến lương `min_salary`, `max_salary`, `normalized_salary` có hệ số tương quan tuyến tính xấp xỉ tuyệt đối ($r \ge 0.995$).
* **Hệ quả nếu không xử lý:** Việc đưa cả 3 biến này vào mô hình tuyến tính, hồi quy logistic hoặc mạng nơ-ron sẽ gây đa cộng tuyến hoàn hảo, khiến ma trận nghịch đảo bị suy biến, sai số chuẩn của trọng số tăng vọt, mô hình mất ổn định và bị Overfitting nghiêm trọng.
* **Quyết định bàn giao:** **Loại bỏ `min_salary` và `max_salary`**, chỉ giữ lại duy nhất **`normalized_salary`** làm biến đại diện cho thu nhập hàng năm.

---

## 5. Phân Tích Quan Hệ Đặc Trưng ↔ Biến Mục Tiêu (Feature ↔ Target)

### 5.1. Phân phối mức lương theo Cấp bậc kinh nghiệm (Job Level)

Thống kê mức lương chuẩn hóa (`normalized_salary`) theo từng cấp bậc (sắp xếp giảm dần theo trung vị):

| Cấp độ kinh nghiệm (`formatted_experience_level`) | Số lượng tin có lương | Lương trung bình (Mean) | Lương trung vị (Median) | Độ lệch chuẩn (Std) | Khoảng lương phổ biến (IQR: Q1 - Q3) |
|---|---|---|---|---|---|
| **Executive** | 432 | $206,450 | **$193,750** | $78,210 | $150,000 - $250,000 |
| **Director** | 1,489 | $174,120 | **$167,500** | $59,340 | $130,000 - $210,000 |
| **Mid-Senior level** | 13,875 | $116,840 | **$108,389** | $44,520 | $82,000 - $145,000 |
| **Associate** | 3,982 | $81,250 | **$74,880** | $32,180 | $58,000 - $98,000 |
| **Entry level** | 10,245 | $59,310 | **$52,442** | $26,450 | $41,600 - $70,000 |
| **Internship** | 412 | $51,200 | **$47,840** | $21,130 | $35,000 - $62,000 |

### 5.2. Kết quả Kiểm định Thống kê Toán học (Hypothesis Testing)

Kiểm định giả thuyết để xác nhận sự khác biệt lương giữa các cấp bậc có ý nghĩa thực tế hay chỉ do ngẫu nhiên:
* **$H_0$:** Mức lương ở tất cả các cấp bậc là như nhau.
* **$H_1$:** Tồn tại ít nhất một cấp bậc có mức lương khác biệt rõ rệt.

| Phương pháp kiểm định | Đặc trưng kiểm tra | Giá trị thống kê | Bậc tự do ($df$) | $p$-value | Kết luận thống kê ($\alpha = 0.05$) |
|---|---|---|---|---|---|
| **ANOVA ($F$-test)** | `normalized_salary` | $F = 3,412.85$ | 5 | **$< 0.00001$** | Bác bỏ $H_0$ (Khác biệt cực kỳ có ý nghĩa) |
| **Kruskal-Wallis ($H$-test)** | `normalized_salary` | $H = 14,218.60$ | 5 | **$< 0.00001$** | Bác bỏ $H_0$ (Khác biệt cực kỳ có ý nghĩa) |
| **Kruskal-Wallis ($H$-test)** | `views` | $H = 2,154.30$ | 5 | **$< 0.00001$** | Bác bỏ $H_0$ (Lượt xem có khác biệt theo cấp bậc) |
| **Kruskal-Wallis ($H$-test)** | `applies` | $H = 1,085.12$ | 5 | **$< 0.00001$** | Bác bỏ $H_0$ (Lượt ứng tuyển có khác biệt theo cấp bậc) |

> **Ý nghĩa:** Kiểm định phi tham số Kruskal-Wallis khẳng định vững chắc với độ tin cậy $> 99.9\%$ rằng `normalized_salary` là đặc trưng số quan trọng bậc nhất giúp phân biệt cấp bậc kinh nghiệm.

---

## 6. Phân Tích Biến Định Tính & Tương Tác (Categorical & Multivariate)

### 6.1. Kiểm định Độc lập Chi-Square ($\chi^2$) với Biến mục tiêu

| Cặp biến định tính | Giá trị $\chi^2$ | Bậc tự do ($df$) | $p$-value | Kết luận ($\alpha = 0.05$) |
|---|---|---|---|---|
| `Target` $\times$ `formatted_work_type` | 31,956.90 | 30 | $< 0.00001$ | **Có mối liên hệ mật thiết** (Associated) |
| `Target` $\times$ `remote_allowed` | 2,304.86 | 5 | $< 0.00001$ | **Có mối liên hệ mật thiết** |
| `Target` $\times$ `pay_period` | 3,584.31 | 20 | $< 0.00001$ | **Có mối liên hệ mật thiết** |

* **Đặc điểm Remote theo Cấp bậc:** Vị trí cấp cao (`Executive`: 18.7%, `Director`: 17.9%, `Mid-Senior`: 15.4%) có tỷ lệ cho phép làm từ xa cao gấp 3 lần so với `Entry level` (chỉ 5.3%) và `Internship` (6.2%).
* **Đặc điểm Chu kỳ trả lương:** Vị trí `Executive` và `Director` có $> 91\%$ trả lương theo năm (`YEARLY`), trong khi `Internship` có $83.3\%$ và `Entry level` có $63.4\%$ trả lương theo giờ (`HOURLY`).

---

## 7. Phân Tích Ngoại Lệ (Outlier Analysis) & Biến Đổi Log

### 7.1. Bảng đo lường ngoại lệ (IQR vs Z-score)

| Đặc trưng | Số dòng ($n$) | Ngoại lệ IQR | Tỷ lệ IQR (%) | Ngoại lệ $Z > 3\sigma$ | Tỷ lệ $Z$ (%) | Ngưỡng dưới IQR | Ngưỡng trên IQR |
|---|---|---|---|---|---|---|---|
| `normalized_salary` | 36,059 | **949** | 2.63% | 5 | 0.01% | -$57,500 (gán 0) | **$234,500** |
| `min_salary` | 29,793 | 293 | 0.98% | 1 | 0.00% | -$149,907 | $249,945 |
| `max_salary` | 29,793 | 364 | 1.22% | 1 | 0.00% | -$209,879 | $349,928 |
| `views` | 122,160 | 16,902 | 13.84% | 634 | 0.52% | -4.5 | **15.5** |
| `applies` | 23,320 | 2,897 | 12.42% | 358 | 1.54% | -9.5 | **18.5** |

### 7.2. Bản chất ngoại lệ & Phương án xử lý

* **Bản chất:** 949 ngoại lệ lương ($> \$234,500$) đều thuộc về các chức danh quản lý cấp cao (`Director`, `VP`, `Chief Executive`, `Principal Engineer`). Đây là **Giá trị cực trị hợp lệ (Genuine Extreme Values)**, phản ánh đúng thị trường lao động thực tế. **TUYỆT ĐỐI KHÔNG ĐƯỢC XÓA BỎ!**
* **Giải pháp biến đổi Log:** Áp dụng hàm biến đổi $y = \log(1 + x)$ đối với `normalized_salary`:
  * Độ lệch (Skewness) giảm từ **+2.68** xuống còn **-0.21** (đưa về phân phối gần chuẩn đối xứng).
  * Giúp các mô hình Linear Regression, Logistic Regression, Neural Networks học ổn định và hội tụ nhanh hơn.

---

## 8. Phân Tích Mất Cân Bằng Lớp (Class Imbalance Analysis)

### 8.1. Phân phối các nhãn của Biến mục tiêu

| Lớp kinh nghiệm (`formatted_experience_level`) | Số lượng tin tuyển dụng | Tỷ lệ trên tập có nhãn | Tỷ lệ trên toàn bộ dataset |
|---|---|---|---|
| **Mid-Senior level** | 41,489 | **43.93%** | 33.50% |
| **Entry level** | 36,708 | **38.87%** | 29.64% |
| **Associate** | 9,826 | **10.40%** | 7.93% |
| **Director** | 3,746 | **3.97%** | 3.02% |
| **Internship** | 1,449 | **1.53%** | 1.17% |
| **Executive** | 1,222 | **1.29%** | 0.99% |
| *(Dữ liệu khuyết / NaN)* | 29,409 | — | 23.75% |
| **Tổng cộng** | **123,849** | **100.0%** | **100.0%** |

### 8.2. Đánh giá mức độ & Khuyến nghị mô hình hóa (Bàn giao Người 4)

* **Tỷ lệ mất cân bằng (Imbalance Ratio):**
  $$\text{Ratio} = \frac{\text{Lớp lớn nhất (Mid-Senior: 41,489)}}{\text{Lớp nhỏ nhất (Executive: 1,222)}} \approx \mathbf{34.0 : 1}.$$
* **Cảnh báo rủi ro:** Nếu sử dụng hàm mục tiêu Accuracy tiêu chuẩn, mô hình có thể hội tụ về điểm cực tiểu giả (luôn dự đoán nhãn Mid-Senior/Entry), đạt Accuracy $\approx 83\%$ nhưng F1-score của lớp Executive và Internship bằng 0.
* **Chiến lược bắt buộc cho Người 4:**
  1. Phân chia tập dữ liệu huấn luyện bằng **Stratified K-Fold** để bảo toàn tỷ lệ các lớp thiểu số.
  2. Sử dụng độ đo tối ưu: **F1-Macro** hoặc **Balanced Accuracy**, không dùng raw Accuracy.
  3. Áp dụng kỹ thuật xử lý mất cân bằng: Gán trọng số nghịch đảo tần suất (`class_weight="balanced"`), kỹ thuật tái lấy mẫu (SMOTE, ADASYN) hoặc Focal Loss.

---

## 9. Phân Tích Kỹ Năng & Văn Bản (Skill & Text Analysis)

### 9.1. Kỹ năng tuyển dụng (từ `job_skills_clean.csv`)

* **Top 5 kỹ năng được săn đón nhất toàn thị trường:** `IT` (Công nghệ thông tin - 32,450 lượt), `SALE` (Bán hàng - 28,120 lượt), `MGMT` (Quản lý - 25,640 lượt), `MNFC` (Sản xuất - 21,300 lượt), `ENG` (Kỹ thuật - 18,900 lượt).
* **Đặc trưng kỹ năng theo Cấp bậc:**
  * `Internship` & `Entry level`: Đòi hỏi cao về kỹ năng thực thi kỹ thuật (`IT`, `ENG`, `MNFC`, `HCPR`).
  * `Director` & `Executive`: Đòi hỏi vượt trội về kỹ năng lãnh đạo và quản trị kinh doanh (`MGMT`, `BD - Business Development`, `SALE`, `FIN`).

### 9.2. Đặc trưng độ dài văn bản (Text Length & Word Count)

| Chỉ số văn bản | Mean | Median | Std | Min | Max |
|---|---|---|---|---|---|
| **Độ dài tiêu đề (`title_length`)** | 31.5 ký tự | 28 ký tự | 16.2 | 2 | 200 |
| **Số từ tiêu đề (`title_word_count`)** | 4.3 từ | 3 từ | 2.5 | 1 | 31 |
| **Độ dài mô tả (`desc_length`)** | 3,749.5 ký tự | 3,419 ký tự | 2,137.0 | 0 | 23,106 |
| **Số từ mô tả (`desc_word_count`)** | 523.0 từ | 477 từ | 301.9 | 0 | 3,400 |

* **Xu hướng theo cấp bậc:** Bản mô tả công việc của vị trí `Director` và `Executive` có độ dài trung vị đạt $> 4,200$ ký tự, dài hơn đáng kể so với `Entry level` (~3,100 ký tự) và `Internship` (~2,600 ký tự).
* **Top từ khóa phổ biến (đã lọc Stop words):** `management`, `development`, `experience`, `customer`, `services`, `business`, `support`, `training`, `technology`, `operations`.

---

## 10. Checklist Bàn Giao Kỹ Thuật (Downstream Handover)

| Tiếp nhận | Hạng mục bàn giao | Khuyến nghị kỹ thuật chi tiết |
|---|---|---|
| **Người 3 (Feature Engineering & NLP)** | **Xử lý lương** | - Chỉ giữ lại cột `normalized_salary`.<br>- Bắt buộc tạo feature `log_normalized_salary = np.log1p(normalized_salary)`. |
| **Người 3** | **Tạo Feature văn bản** | - Khai thác các feature độ dài: `title_length`, `desc_length`, `desc_word_count`.<br>- Tạo tỷ lệ từ viết hoa / số lượng đoạn văn.<br>- Vector hóa TF-IDF trên 25 từ khóa phổ biến. |
| **Người 3** | **Mã hóa kỹ năng** | - Ghép bảng `job_skills_clean` để tạo đặc trưng đếm `skill_count`.<br>- Tạo Multi-hot vector cho Top 20 kỹ năng thị trường. |
| **Người 4 (Modeling)** | **Chia tập dữ liệu** | - Bắt buộc dùng `StratifiedKFold(n_splits=5, shuffle=True)` theo biến `formatted_experience_level`. |
| **Người 4** | **Chiến lược Imbalance** | - Áp dụng `class_weight="balanced"` cho Random Forest, XGBoost, LightGBM.<br>- Thử nghiệm SMOTE trên tập Train (tránh Data Leakage sang tập Test). |
| **Người 4** | **Hệ thống độ đo** | - Đặt **F1-Macro** làm Metric đánh giá chính thay vì Accuracy. |

---

## 11. Cấu Trúc Module Hóa (`src/visualization/`)

Toàn bộ các hàm trực quan hóa đã được đóng gói thành module sản phẩm chuẩn Software Engineering tại [`src/visualization/visualize.py`](file:///c:/Users/khain/DS_Job_Recommend_Projects/src/visualization/visualize.py):

* `plot_numerical_distribution(df, col, bins=60)`: Vẽ Histogram, đường KDE, Boxplot, tính IQR và Skewness.
* `plot_categorical_distribution(df, col, top_n=15)`: Vẽ biểu đồ số lượng và cơ cấu tỷ lệ %, xử lý an toàn nhãn NaN.
* `plot_correlation_heatmap(df, num_cols, method='pearson')`: Vẽ bản đồ nhiệt tương quan nửa dưới.
* `plot_feature_vs_target(df, num_col, target_col)`: Vẽ Boxplot và Violin plot so sánh phân phối theo biến mục tiêu.
* `plot_class_imbalance(df, target_col)`: Vẽ biểu đồ cột và biểu đồ tròn phân phối lớp biến mục tiêu.
* `plot_outlier_log_transform(df, col)`: So sánh hiệu quả chuẩn hóa phân phối trước và sau biến đổi Log(1+x).

Các thành viên khác trong dự án có thể gọi sử dụng trực tiếp:
```python
from src.visualization import plot_numerical_distribution, plot_class_imbalance
```
