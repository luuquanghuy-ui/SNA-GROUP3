# E1 – Học biểu diễn đa nhánh (Multi-Branch Representation Learning)

## 1. Tổng quan

E1 xây dựng mô hình phát hiện tài khoản Bot bằng phương pháp học biểu diễn đa nhánh kết hợp với học tương phản có giám sát (Supervised Contrastive Learning).

Mô hình sử dụng 4 nhánh đầu vào:

| Nhánh đầu vào | Số chiều |
|---|---:|
| Biểu diễn nội dung tweet | 384 |
| Biểu diễn văn bản hồ sơ | 384 |
| Đặc trưng số và Boolean của hồ sơ | 15 |
| Biểu diễn tổng hợp từ tài khoản lân cận | 768 |

Đặc trưng từ 4 nhánh được đưa qua các lớp chiếu, sau đó kết hợp để phân loại tài khoản thành Human hoặc Bot.

## 2. Mô hình mã hóa văn bản

- Mô hình: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Số chiều embedding: 384.
- Embedding được tạo trước và lưu vào bộ nhớ đệm.
- MiniLM được giữ cố định, không được huấn luyện lại trong quá trình huấn luyện E1.

## 3. Kiến trúc mô hình

- Phương pháp: Học biểu diễn đa nhánh.
- Hàm mất mát: Cross-Entropy kết hợp Supervised Contrastive Loss.
- Đầu ra phân loại: 2 lớp.
- Đầu ra biểu diễn: vector chuẩn hóa có 128 chiều.

Quy ước nhãn:
- Human = 0
- Bot = 1

## 4. Cấu hình huấn luyện

- Bộ tối ưu: AdamW.
- Tốc độ học: 0.0002.
- Kích thước batch: 4.
- Random seed: 42.
- Batch cân bằng gồm 2 Human và 2 Bot.

## 5. Cấu trúc thư mục bàn giao

```text
e1_handoff/
├── checkpoints/
│   ├── best_model.pt
│   └── training_history.json
├── model/
│   ├── e1_multibranch.py
│   ├── e1_losses.py
│   ├── train_e1.py
│   ├── evaluate_test.py
│   └── prepare_dataloader.py
├── preprocessing/
│   ├── feature_config.json
│   └── profile_numeric_scaler.joblib
├── evaluation/
│   ├── test_metrics.json
│   └── test_predictions.npz
└── README.md
```

## 6. Kết quả đánh giá trên tập Test

| Chỉ số | Kết quả |
|---|---:|
| Accuracy | 0.84 |
| Macro F1-score | 0.84 |
| Weighted F1-score | 0.84 |

Kết quả chi tiết được lưu trong `evaluation/test_metrics.json`.

## 7. Hướng dẫn bàn giao cho E2

Các thành phần E2 cần sử dụng:

- `checkpoints/best_model.pt`: trọng số mô hình đã huấn luyện.
- `model/e1_multibranch.py`: định nghĩa kiến trúc mô hình.
- `preprocessing/feature_config.json`: cấu hình đặc trưng đầu vào.
- `preprocessing/profile_numeric_scaler.joblib`: bộ chuẩn hóa đặc trưng số.
- `model/e1_losses.py`: định nghĩa hàm mất mát.

**Lưu ý:** Khi sử dụng checkpoint, cần giữ đúng thứ tự và số chiều của 4 nhánh đầu vào. Đầu ra biểu diễn 128 chiều (`z`) có thể được sử dụng cho các bước phân tích hoặc đánh giá tiếp theo của E2.

Các file embedding cache và dữ liệu trung gian không nằm trong danh sách file bàn giao mã nguồn và checkpoint.