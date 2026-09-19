# Báo Cáo Phân Tích Khám Phá Dữ Liệu (EDA)
**Dự án:** DS Job Recommend (Hệ thống Khuyến nghị & Dự đoán Cấp bậc Công việc)
**File dữ liệu:** `data/processed/postings_clean.csv`
**Biến mục tiêu (Target):** `formatted_experience_level`

---

## 1. Tổng quan Dữ liệu (Descriptive Statistics)
- Dữ liệu chứa thông tin về các bài đăng tuyển dụng, bao gồm các đặc trưng về lương (`min_salary`, `med_salary`, `max_salary`, `normalized_salary`), mức độ tương tác (`views`, `applies`) và thông tin mô tả công việc.
- Sử dụng hàm `info()` và `describe()` để nắm bắt các tham số thống kê cơ bản như trung bình, trung vị, độ lệch chuẩn.

## 2. Phân tích Missing Value & Duplicate
- **Giá trị khuyết thiếu (Missing Values):** Dữ liệu có tỷ lệ khuyết thiếu khá cao ở một số cột, đặc biệt là các cột liên quan đến mức lương (ví dụ: `med_salary`, `applies`).
- **Dữ liệu trùng lặp (Duplicates):** Đã kiểm tra tỷ lệ trùng lặp để đảm bảo tính toàn vẹn của dữ liệu đầu vào.

## 3. Phân phối Dữ liệu Số (Numerical Distribution)
Thông qua biểu đồ **Histplot kết hợp đường KDE**, phân phối của các biến số chính (như lương chuẩn hóa, lượt xem, lượt ứng tuyển) có các đặc điểm:
- **Lệch phải (Right-skewed):** Phần lớn tin tuyển dụng có mức lương và lượt tương tác ở mức trung bình - thấp, và một số ít tin có mức lương/tương tác rất cao kéo dài về bên phải.
- **Biện pháp xử lý:** Trong bước tiền xử lý mô hình, có thể cân nhắc dùng phép biến đổi Log (Log-transformation) để đưa phân phối về dạng chuẩn hơn.

## 4. Phát hiện Ngoại lệ (Outliers Detection)
Sử dụng biểu đồ **Boxplot (chia subplots)** để nhận diện ngoại lệ:
- Các cột về lương và tương tác có chứa nhiều điểm dữ liệu nằm ngoài ranh giới trên (Q3 + 1.5*IQR).
- Mức lương cực cao này tuy là ngoại lệ về mặt toán học nhưng phản ánh đúng thực tế các vị trí cấp quản lý/giám đốc (Director, Executive), do đó cần xử lý cẩn thận (không nên xóa bỏ tùy tiện).

## 5. Phân tích Tương quan (Correlation Matrix)
Sử dụng **Heatmap** để đo lường mức độ tương quan giữa các biến số:
- Nhóm biến về lương (`min_salary`, `max_salary`, `normalized_salary`) có hệ số tương quan rất gần 1.
- **Đa cộng tuyến (Multicollinearity):** Hiện tượng này cho thấy các cột này mang thông tin trùng lặp. Đề xuất chỉ giữ lại một cột (như `normalized_salary`) để đưa vào mô hình nhằm tránh nhiễu và quá khớp (overfitting).

## 6. Mối quan hệ giữa các Đặc trưng (Feature Relationship)
Dùng **Scatter Matrix** để trực quan hóa mối quan hệ phân tán giữa nhiều biến cùng lúc.
- Lượt xem (`views`) và lượt ứng tuyển (`applies`) có tương quan thuận chiều với nhau (nhiều view thì thường nhiều apply).
- Các biến lương không cho thấy tương quan tuyến tính rõ rệt với lượt xem/ứng tuyển.

## 7. Phân tích Biến Mục Tiêu (Target Variable Analysis)
Biểu đồ **Countplot** cho biến `formatted_experience_level` chỉ ra:
- **Mất cân bằng lớp (Class Imbalance):** Các nhóm `Mid-Senior level` và `Entry level` chiếm tỷ lệ áp đảo so với các nhóm `Executive` hay `Internship`.
- **Khuyến nghị:** Cần sử dụng các kỹ thuật cân bằng dữ liệu (như SMOTE hoặc gán class_weight) trong quá trình huấn luyện mô hình để không bị thiên lệch (bias) về các lớp chiếm số đông.

## 8. Phân tích Lương theo Cấp Bậc Kinh Nghiệm
Dùng **Boxplot** so sánh phân phối `normalized_salary` giữa các cấp bậc (Job Levels):
- Mức lương tăng dần theo cấp bậc rất rõ ràng (Executive > Director > Mid-Senior > Associate > Entry > Internship).
- Điều này chứng minh rằng mức lương chuẩn hóa là một đặc trưng cực kỳ quan trọng và có sức mạnh phân loại cao để dự đoán biến mục tiêu `formatted_experience_level`.
