# BÁO CÁO KIỂM ĐỊNH E1 – FOX8-23

## 1. Tổng quan

- **Mục tiêu:** Phân loại tài khoản Bot/Human và tạo biểu diễn đặc trưng phục vụ các thí nghiệm tiếp theo.
- **Dữ liệu:** Fox8-23.
- **Random seed:** 42.
- **Thiết bị:** CPU.
- **Tỷ lệ chia dữ liệu:** Train 70%, Validation 10%, Test 20%.

## 2. Tiền xử lý dữ liệu

Tweet được tổng hợp theo từng tài khoản trước khi chuyển thành đặc trưng TF-IDF.

| Tham số | Giá trị |
|---|---|
| Số tweet tối đa mỗi tài khoản | 30 |
| Độ dài văn bản tối đa | 5.000 ký tự |
| N-gram | Unigram và bigram |
| Số đặc trưng tối đa | 10.000 |
| `min_df` | 2 |
| `sublinear_tf` | True |
| Kiểu dữ liệu | `float32` |

Cấu hình TF-IDF đã được đối chiếu với file cấu hình E1. Các kiểm tra về tỷ lệ chia dữ liệu và trùng lặp tài khoản giữa các tập cũng đã được thực hiện.

## 3. Cấu hình mô hình

| Mô hình | Cấu hình chính |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"` |
| Random Forest | 200 cây, `class_weight="balanced"` |
| MLP | Hidden dimension 256, embedding dimension 128, tối đa 20 epochs |

Encoder của MLP chuyển đổi vector TF-IDF 10.000 chiều thành biểu diễn 128 chiều.

## 4. Kết quả đánh giá trên tập kiểm thử (Test Results)

Tập kiểm thử gồm **456 tài khoản**, trong đó có 228 tài khoản Bot và 228 tài khoản Human.

### 4.1. Kết quả các mô hình

| Mô hình | Accuracy | Precision (Bot) | Recall (Bot) | F1-score (Bot) | Macro F1 | Average Precision (Bot) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 100% | 100% | 100% | 1.0000 | 1.0000 | 1.0000 |
| MLP | 100% | 100% | 100% | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 99,78% | 99,56% | 100% | 0.9978 | 0.9978 | 1.0000 |

### 4.2. Ma trận nhầm lẫn (Confusion Matrix)

| Mô hình | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 228 | 0 | 0 | 228 |
| MLP | 228 | 0 | 0 | 228 |
| Random Forest | 227 | 1 | 0 | 228 |

Trong đó:
- **TN (True Negative):** Số tài khoản Human được dự đoán đúng.
- **FP (False Positive):** Số tài khoản Human bị dự đoán nhầm thành Bot.
- **FN (False Negative):** Số tài khoản Bot bị dự đoán nhầm thành Human.
- **TP (True Positive):** Số tài khoản Bot được dự đoán đúng.


## 5. Kiểm định artifact

| Hạng mục | Kết quả |
|---|---|
| Metrics và predictions | Đạt |
| Cấu hình TF-IDF | Đạt |
| Cấu trúc encoder checkpoint | Đạt |
| Trọng số encoder giữa hai checkpoint | Khớp hoàn toàn |
| Tỷ lệ chia dữ liệu 70/10/20 | Đã kiểm tra |
| Trùng lặp tài khoản giữa các tập | Đã kiểm tra |

**Lưu ý:** Vectorizer được lưu bằng scikit-learn 1.6.1, trong khi môi trường kiểm tra sử dụng phiên bản 1.8.0. Cần lưu ý khác biệt phiên bản khi tải file ở môi trường khác.

## 6. Hạn chế

- Nhãn Bot và Human có liên hệ với các nguồn thu thập dữ liệu khác nhau. Mô hình có thể học đặc điểm riêng của nguồn dữ liệu thay vì đặc điểm chung của bot.
- Kết quả cao trên tập test Fox8-23 chưa chứng minh khả năng tổng quát hóa sang bộ dữ liệu khác.
- Encoder sử dụng đặc trưng TF-IDF từ văn bản tweet, không phải biểu diễn hành vi đa phương thức.

## 7. Kết luận

Các kiểm tra về metrics, predictions, cấu hình TF-IDF và tính nhất quán của encoder checkpoint đều đạt yêu cầu. Việc chia dữ liệu và kiểm tra trùng lặp tài khoản đã được xác nhận riêng.

E1 có thể được sử dụng làm baseline và bước trích xuất biểu diễn cho các thí nghiệm tiếp theo. Tuy nhiên, cần thận trọng khi diễn giải kết quả do nguy cơ **source-label confounding**. Khả năng tổng quát hóa sang dữ liệu khác vẫn chưa được xác lập.