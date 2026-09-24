# Báo Cáo Phân Tích Khám Phá Dữ Liệu (Exploratory Data Analysis Report)
**Project:** LinkedIn Job Postings — Job Level Analysis & Prediction  
**Branch:** `feature/visualization`  
**Vai trò:** Data Analyst / Visualization Engineer (NGƯỜI 2)  

---

## 1. Tổng quan Quá trình Khám phá Dữ liệu (Executive Summary)

Dữ liệu phân tích được tiếp nhận từ quá trình làm sạch (`postings_clean.csv`, `job_skills_clean.csv`) gồm **123,849 dòng** và **31 cột**. Quá trình EDA được thực hiện nhằm cung cấp cái nhìn toàn diện về phân phối, mối quan hệ giữa các biến số, phân tích văn bản và kỹ năng. 

### Các điểm sáng của quá trình phân tích:
1. **Lương là đặc trưng cực kỳ quan trọng:** Trung vị mức lương (`normalized_salary`) tăng tuyến tính và rõ rệt qua các cấp bậc kinh nghiệm, cho thấy đây là một "hard feature" giúp mô hình phân loại tốt (từ Internship thấp nhất đến Executive cao nhất).
2. **Khám phá Đa cộng tuyến:** Phát hiện sự tương quan gần như tuyệt đối ($r \approx 1.0$) giữa các biến `min_salary`, `med_salary`, `max_salary` và `normalized_salary`.
3. **Mất cân bằng dữ liệu mục tiêu nghiêm trọng:** Tỷ lệ chênh lệch giữa lớp lớn nhất (Mid-Senior) và lớp nhỏ nhất (Executive) lên tới 34:1.
4. **Phân tích Văn bản & Kỹ năng (Text & Skills):** Có sự khác biệt rõ rệt về độ dài mô tả công việc và bộ kỹ năng yêu cầu giữa nhóm Entry-level (thiên về kỹ thuật) và nhóm Executive (thiên về quản lý, kinh doanh).

---

## 2. Chi tiết Phân tích Đặc trưng & Biến Mục Tiêu (Feature & Target Analysis)

Quá trình phân tích được chia thành các hướng tiếp cận chi tiết, đánh giá tác động của từng nhóm đặc trưng lên biến mục tiêu `formatted_experience_level`.

### 2.1 Phân tích Đơn biến (Univariate Analysis)
- **Biến số (Numerical):** Các biến `normalized_salary`, `views`, và `applies` đều có phân phối **lệch phải rất mạnh (right-skewed)**. Hầu hết các tin tuyển dụng tập trung ở mức lương, tương tác trung bình - thấp, và xuất hiện một phần đuôi dài (long-tail) ở các vị trí cấp cao.
- **Biến phân loại (Categorical):** 
  - Khối lượng công việc chủ yếu là **Full-time** (chiếm ~81%).
  - Lương theo năm (**YEARLY**) chiếm ưu thế (~65%), theo giờ (**HOURLY**) chiếm khoảng 33%.
  - Chỉ có **~12%** vị trí cho phép **Remote**, còn lại là On-site hoặc Hybrid.

### 2.2 Tương quan & Đa cộng tuyến (Correlation & Multicollinearity)
- Sử dụng **Ma trận Tương quan (Heatmap)** để phân tích các biến số.
- Các cột `min_salary`, `max_salary`, và `normalized_salary` có độ tương quan rất cao. 
- **Quyết định xử lý:** Để tránh hiện tượng đa cộng tuyến (Multicollinearity) làm nhiễu mô hình, chỉ nên giữ lại cột `normalized_salary` cho quá trình huấn luyện.

### 2.3 Phân tích Biến Mục Tiêu (Feature ↔ Target Analysis)
- Các biểu đồ **Boxplot** và **Barplot** cho thấy sự chênh lệch rõ ràng: Lương trung bình của Executive có thể gấp 3-4 lần so với Internship.
- Phân tích Crosstab chỉ ra rằng: Các vị trí cấp cao (Director, Executive) có tỷ lệ Full-time và cho phép làm Remote cao hơn hẳn so với các vị trí mới vào nghề.

### 2.4 Phân tích Phân phối Lệch & Ngoại lệ (Outliers & Skewness)
- **Phát hiện:** Subplots Boxplot phát hiện rất nhiều giá trị ngoại lệ trên các cột lương. Tuy nhiên, đây là **Ngoại lệ hợp lệ (Genuine Extreme Values)** phản ánh lương của các vị trí cấp cao, hoàn toàn không phải lỗi dữ liệu.
- **Xử lý:** Áp dụng phép biến đổi Logarit (`np.log1p(x)`) cho các biến số lệch phải. Độ lệch (Skewness) của lương giảm mạnh từ $2.68$ xuống gần mức $0$, đưa phân phối về dạng hình chuông (chuẩn) giúp các thuật toán Machine Learning hội tụ nhanh và tốt hơn.

---

## 3. Phân Tích Thông Tin Phi Cấu Trúc (Unstructured Data Analysis)

Để khai thác tối đa thông tin từ tin tuyển dụng, quá trình EDA đã phân tích sâu vào các trường dữ liệu phi cấu trúc như kỹ năng và mô tả văn bản:

### 3.1 Phân tích Kỹ năng (Skill Analysis)
- Kết hợp với bảng `job_skills_clean.csv`, xác định được Top 3 nhóm kỹ năng phổ biến nhất trên thị trường là **IT (Công nghệ thông tin)**, **SALE (Bán hàng)**, và **MGMT (Quản lý)**.
- **Insight thú vị:** Nhóm Entry/Internship tập trung cực độ vào các kỹ năng công cụ/chuyên môn (IT, ENG), trong khi nhóm Director/Executive yêu cầu các kỹ năng vĩ mô (MGMT, SALE, FIN, MKTG).

### 3.2 Phân tích Đặc trưng Văn bản (Text Analysis)
- Kỹ thuật Feature Extraction cơ bản được áp dụng để tạo ra 3 biến mới: `title_length`, `desc_length`, và `desc_word_count`.
- **Đánh giá:** Boxplot cho thấy vị trí cấp bậc càng cao thì xu hướng viết mô tả công việc (`desc_length`) càng dài và chi tiết hơn. Đây là các tính năng (features) tiềm năng giúp mô hình dự đoán chính xác hơn.

---

## 4. Đánh Giá Chất Lượng Đặc Trưng (Feature Evaluation Summary)

Bảng điều khiển tổng hợp (4-Panel Overview) đã được xây dựng tại phần 11 để đánh giá tổng thể chất lượng của các biến số trước khi đưa vào mô hình học máy:

| Tiêu chí | Nội dung Đánh giá | Mức độ Cảnh báo (Theo màu biểu đồ) | Đề xuất Xử lý (Cho NGƯỜI 3) |
|---|---|---|---|
| **1. Missing Ratio** | Tỷ lệ % giá trị rỗng (NaN) | Lương thiếu khoảng 56%, Views thiếu trung bình. | Cần áp dụng kỹ thuật Imputation (ví dụ: Median) hoặc mã hóa là một giá trị riêng. |
| **2. Skewness** | Độ lệch phân phối | `applies` và `views` có độ lệch lớn (Cảnh báo Đỏ). | Bắt buộc thực hiện Log-Transform để nén các giá trị lệch. |
| **3. Outliers** | Số lượng ngoại lệ (IQR) | Hàng chục ngàn giá trị nằm ngoài ranh giới 1.5*IQR. | Khuyến nghị sử dụng Robust Scaler thay vì Standard Scaler để tránh bị nhiễu. |
| **4. Target Correlation** | Mức độ tương quan với Target | `normalized_salary` tương quan tốt (Xanh). `views` / `applies` tương quan yếu. | Giữ nguyên các biến lương, cân nhắc loại bỏ các tương tác yếu nếu gây nhiễu mô hình. |

---

## 5. Kết Luận & Định Hướng Mô Hình Hóa (Modeling Recommendations)

### Cảnh báo nghiêm trọng: Mất cân bằng dữ liệu (Class Imbalance)
- Dữ liệu mục tiêu bị mất cân bằng trầm trọng với tỷ lệ **34:1**. Lớp đa số (Mid-Senior) sẽ áp đảo quá trình học của mô hình, khiến mô hình thiên vị lớp này và bỏ qua các lớp thiểu số (Executive, Internship).

### Bàn giao & Khuyến nghị cho Nhóm Feature Engineering & Modeling (NGƯỜI 3 & 4):
Để mô hình phân loại đạt chất lượng thực tiễn, nhóm Modeling cần áp dụng nghiêm ngặt các quy tắc sau:
1. **Xử lý Mất cân bằng:** Bắt buộc áp dụng phương pháp như **SMOTE** (Synthetic Minority Over-sampling Technique) để sinh thêm mẫu cho lớp thiểu số, hoặc thiết lập tham số `class_weight='balanced'`.
2. **Chiến lược Chia dữ liệu:** Bắt buộc sử dụng kỹ thuật **Stratified K-Fold** khi chia tập Train/Test để giữ nguyên tỷ lệ phân phối gốc của các cấp bậc.
3. **Đánh giá Hiệu suất:** Tuyệt đối không dùng chỉ số `Accuracy` (độ chính xác tổng thể). Thay vào đó, lấy **F1-Macro** hoặc **Balanced Accuracy** làm hệ quy chiếu để đánh giá khả năng dự đoán công bằng trên tất cả 6 cấp bậc công việc.
