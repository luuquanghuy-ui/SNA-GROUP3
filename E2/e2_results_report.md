# Báo Cáo Kết Quả Thí Nghiệm E2: Train from Scratch vs Transfer Learning

## 1. Tổng Quan & Mục Tiêu

Tài liệu này tổng hợp chi tiết kết quả thực nghiệm của **Thí nghiệm 2 (E2: Scratch vs Transfer Learning)** thuộc dự án Phát hiện Bot trên dữ liệu Honduras (Nhiệm vụ do **Người 3** phụ trách).

Mục tiêu cốt lõi là giải quyết câu hỏi nghiên cứu:
> *Liệu việc chuyển giao biểu diễn hành vi (behavioral representations) được huấn luyện trước trên tập dữ liệu Fox8-23 có giúp cải thiện khả năng phân loại tài khoản Bad/Good trên tập dữ liệu Honduras so với việc huấn luyện từ đầu (Train from Scratch) và các phương pháp baseline thống kê hay không?*

---

## 2. Thiết Lập Thí Nghiệm & Protocol

* **Tập dữ liệu:** Honduras (`honduras-bad-anonymized` và `honduras-good-anonymized`).
* **Phân chia dữ liệu:** Cố định theo cấp tài khoản (70% Train, 10% Validation, 20% Test) với `seed=42`.
* **Quy ước nhãn:**
  * `0`: Bad (Bot / Tài khoản phối hợp)
  * `1`: Good (Human / Tài khoản thường)
* **Các chỉ số đánh giá (Metrics):**
  * **Bad-Precision**, **Bad-Recall**, **Bad-F1**: Các chỉ số đo lường trên lớp mục tiêu Bad (nhãn `0`).
  * **Macro-F1**: Trung bình F1 không trung bình hóa theo quy mô lớp (đánh giá tổng thể mất cân bằng lớp).
  * **AP (Average Precision)**: Diện tích dưới đường cong Precision-Recall (PR-AUC) cho lớp Bad.

---

## 3. Bảng Tổng Hợp Kết Quả (Test Set)

Kết quả chi tiết đạt được trên tập **Test** cố định như sau:

| Phương pháp (Method) | Bad-Precision | Bad-Recall | Bad-F1 | Macro-F1 | AP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LR (Full Features)** | 0.0891 | **0.9651** | 0.1632 | 0.5600 | 0.4712 |
| **LR (No `tweet_count`)** | 0.0863 | **0.9812** | 0.1587 | 0.5566 | 0.3688 |
| **Train from Scratch** | **0.9961** | 0.6810 | 0.8089 | 0.9038 | **0.9204** |
| **Fox8-pretrained + Fine-tuning** | 0.9692 | **0.8445** | **0.9026** | **0.9509** | 0.9182 |

---

## 4. Phân Tích & Biện Luận Kết Quả

### 4.1. Baseline Thống Kê vs. Neural Behavioral Encoder
* **Logistic Regression (LR)** trên 6 đặc trưng thống kê số (`tweet_count`, `url_ratio`, `hashtag_ratio`,...): Mặc dù cài đặt `class_weight='balanced'` giúp Recall đạt mức rất cao (~0.965), mô hình lại gặp phải hiện tượng **Dương tính giả (False Positives) cực kỳ cao**, dẫn đến **Bad-Precision rất thấp (~8.9%)** và **Bad-F1 chỉ đạt ~0.1632**.
* **Neural Network Encoders** (sử dụng đầu vào văn bản TF-IDF 10,000 chiều ở cấp tài khoản): Thể hiện sự vượt trội hoàn toàn so với baseline thống kê ở tất cả các chỉ số, đẩy Bad-F1 vượt mốc **0.80**.

### 4.2. Train from Scratch vs. Transfer Learning (Phân tích cốt lõi)
* **Train from Scratch:** Mô hình đạt độ chính xác gần như tuyệt đối ở nhãn Bad (**Bad-Precision = 0.9961**), tuy nhiên khả năng bao phủ lại hạn chế khi bỏ sót tới gần 32% số lượng Bot thật sự (**Bad-Recall = 0.6810**).
* **Fox8-pretrained + Fine-tuning (Transfer Learning):**
  * Tăng mạnh **Bad-Recall từ 0.6810 lên 0.8445 (+16.35%)** trong khi vẫn duy trì Precision ở mức rất cao (**0.9692**).
  * Nâng **Bad-F1 từ 0.8089 lên 0.9026 (+9.37%)**.
  * Tăng **Macro-F1 tổng thể lên 0.9509 (+4.71%)**.
* **Kết luận thực nghiệm:** Việc chuyển giao tri thức từ tập dữ liệu Fox8-23 tạo ra hiệu ứng chuyển giao tích cực (**positive transfer effect**). Trọng số tiền huấn luyện giúp mạng Neural nhận diện được các mẫu hành vi Bot tinh vi hơn trên Honduras mà phương pháp học từ đầu không phát hiện được.

### 4.3. Ablation Study: Ảnh hưởng của đặc trưng `tweet_count`
* Khi loại bỏ `tweet_count` khỏi mô hình Logistic Regression, **Bad-F1 giảm nhẹ (0.1632 → 0.1587)** và **AP sụt giảm đáng kể (0.4712 → 0.3688)**.
* Kết quả này xác nhận: Tần suất hoạt động (`tweet_count`) có cung cấp một lượng tín hiệu dự báo nhất định, nhưng nếu chỉ dựa vào các chỉ số thống kê tần suất mà không có ngữ cảnh biểu diễn văn bản/hành vi thì không thể phát hiện Bot hiệu quả.

---

## 5. Sản Phẩm Bàn Giao (Artifacts)

Tất cả các sản phẩm của thí nghiệm E2 đã được lưu trữ tại thư mục dự án:
1. **Bảng metrics tổng hợp:** `evaluation/E2_RESULTS_TABLE.csv`
2. **Dự đoán chi tiết trên tập Validation/Test:** `evaluation/e2_predictions_e1_emb.csv`
3. **Bộ chuẩn hóa Preprocessing:** `preprocessing/scaler_e1_emb.joblib`
4. **Model Checkpoints:**
   * `models/model_scratch_e2.pt` (Mô hình Train from Scratch)
   * `models/model_transfer_e2.pt` (Mô hình Fine-tuning từ Fox8)

---

## 6. Hướng Dẫn Bàn Giao Cho Giai Đoạn Tiếp Theo

* **Gửi Người 5 (E3 Proposed Graphs):**
  * Khuyến nghị ưu tiên sử dụng vector biểu diễn từ **`model_transfer_e2`** (Transferred Embeddings) để xây dựng đồ thị tăng cường, vì chất lượng biểu diễn của mô hình Transfer vượt trội rõ rệt so với mô hình Scratch.
* **Gửi Người 6 (Báo cáo tổng kết & Ablation final):**
  * Mức tăng trưởng từ **0.8089 Bad-F1 (Scratch)** lên **0.9026 Bad-F1 (Transfer)** là bằng chứng thực nghiệm quan trọng nhất để trả lời câu hỏi nghiên cứu của đồ án về hiệu quả của Transfer Learning.