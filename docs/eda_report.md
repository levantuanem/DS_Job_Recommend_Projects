# Báo Cáo Phân Tích Khám Phá Dữ Liệu (EDA Report)
**Dự án:** DS Job Recommend  
**File dữ liệu:** `data/processed/postings_clean.csv`  
**Biến mục tiêu:** `formatted_experience_level`

---

## 1. Environment Setup & Data Loading
- Dataset gồm 123,849 dòng × 31 cột chứa thông tin tuyển dụng LinkedIn.
- Sử dụng `info()`, `describe()` để nắm bắt thống kê cơ bản.
- Kiểm tra Missing Values: nhiều cột có tỷ lệ khuyết cao (salary, applies).
- Kiểm tra Duplicate: tỷ lệ trùng lặp thấp.

## 2. Univariate Analysis (Numerical & Categorical)
### Numerical (Histplot + KDE)
- Các cột `normalized_salary`, `views`, `applies` đều có phân phối **lệch phải (right-skewed)**.
- Phần lớn tin tuyển dụng có mức lương và tương tác ở mức trung bình - thấp.

### Categorical (Countplot)
- **`formatted_work_type`:** Full-time chiếm đa số (~81%), tiếp theo Contract, Part-time.
- **`remote_allowed`:** Chỉ ~12% cho phép Remote, còn lại là Onsite/Hybrid.
- **`pay_period`:** Chủ yếu trả lương YEARLY (~65%) và HOURLY (~33%).

## 3. Feature ↔ Feature Analysis (Correlation & Multicollinearity)
- Sử dụng **Heatmap** để đo tương quan giữa các biến số.
- **Phát hiện đa cộng tuyến:** `min_salary`, `max_salary`, `normalized_salary` có tương quan rất gần 1 → thông tin trùng lặp → chỉ nên giữ lại `normalized_salary`.
- `views` và `applies` có tương quan thuận vừa phải (~0.65).

## 4. Feature ↔ Target Analysis (Group Comparison)
- Sử dụng **Boxplot** và **Barplot** so sánh lương theo cấp bậc.
- Lương tăng rõ rệt theo thứ tự: Executive > Director > Mid-Senior > Associate > Entry > Internship.
- Chứng tỏ `normalized_salary` là đặc trưng quan trọng để dự đoán biến mục tiêu.

## 5. Categorical ↔ Numerical Analysis
- **Work Type vs Salary (Boxplot):** Mức lương khác nhau rõ rệt giữa các loại hình công việc.
- **Remote vs Salary (Boxplot):** Vị trí cho phép Remote thường có mức lương cao hơn so với Onsite.

## 6. Categorical ↔ Categorical Analysis (Crosstab)
- Sử dụng **Crosstab** và **Stacked Bar Chart** để xem tỷ lệ phần trăm.
- **Work Type theo Level:** Executive/Director hầu hết là Full-time; Internship có tỷ lệ Part-time cao hơn.
- **Remote theo Level:** Vị trí cấp cao (Executive, Director) có tỷ lệ Remote cao hơn Entry/Internship.

## 7. Multivariate Analysis (Scatter Matrix)
- Sử dụng **Scatter Matrix** để xem quan hệ giữa nhiều biến số cùng lúc.
- Views và Applies có xu hướng tăng cùng nhau.
- Lương không có tương quan tuyến tính rõ rệt với lượt xem/ứng tuyển.

## 8. Outlier Analysis & Log Transformation
- **Boxplot subplots:** Các biến salary có nhiều outlier (lương cấp cao Director/Executive).
- **Log Transformation:** Áp dụng `log(1+x)` giúp giảm Skewness đáng kể, đưa phân phối về dạng gần chuẩn hơn → giúp mô hình học tốt hơn.

## 9. Class Imbalance Analysis & Modeling Recommendations
- **Mất cân bằng lớp:** Mid-Senior và Entry chiếm đa số; Executive và Internship rất ít.
- **Tỷ lệ mất cân bằng:** khoảng 34:1.
- **Khuyến nghị:**
  1. Dùng Stratified K-Fold để bảo toàn tỷ lệ lớp.
  2. Dùng F1-Macro hoặc Balanced Accuracy thay vì Accuracy.
  3. Áp dụng SMOTE hoặc class_weight="balanced".

## 10. Skill Analysis
- Từ file `job_skills_clean.csv`, thống kê **Top 15 kỹ năng phổ biến nhất**.
- Phân tích kỹ năng theo từng cấp bậc kinh nghiệm:
  - Entry/Internship: đòi hỏi nhiều kỹ năng kỹ thuật (IT, ENG).
  - Director/Executive: đòi hỏi nhiều kỹ năng quản lý (MGMT, SALE, FIN).

## 11. Text Analysis
- Tính toán độ dài tiêu đề (`title_length`), độ dài mô tả (`desc_length`), số từ mô tả (`desc_word_count`).
- Dùng **Histplot** để vẽ phân phối độ dài văn bản.
- Dùng **Boxplot** so sánh độ dài mô tả theo cấp bậc: vị trí cấp cao có mô tả dài hơn.

## 12. Business Insights & Executive Dashboard
- Tổng hợp các phát hiện chính:
  1. Mid-Senior level được tuyển nhiều nhất, Executive ít nhất.
  2. Lương trung vị tăng dần theo cấp bậc.
  3. Đa cộng tuyến giữa các cột salary → giữ lại normalized_salary.
  4. Class imbalance nghiêm trọng → bắt buộc dùng SMOTE/class_weight.
- **Executive Dashboard** gồm 4 biểu đồ tổng hợp trên 1 hình: phân bố target, salary by level, correlation heatmap, views vs applies scatter.
