# SNA-GROUP3
bot detection on Fox8-23, Bad/Good classification and coordination graph analysis on Honduras.

## Cấu trúc

| Thư mục | Nội dung |
| --- | --- |
| `data/preprocessing/` | Khảo sát, chuẩn hóa, manifest, split và audit duplicate |
| `data/validation/` | Xác minh nhãn Honduras và tạo feature cấp tài khoản |
| `E1/` | Bot/Human baseline trên Fox8-23 |
| `E2/` | Bad/Good baseline trên Honduras |
| `E3/` | Graph hành vi và phân tích cộng đồng |
| [evaluation](evaluation/README.md) | Đánh giá, tổng hợp và kiểm tra tái lập |

Các thư mục data và E1/E2/E3 hiện dùng `.gitkeep` để giữ cấu trúc; code chưa được tích hợp vào các thư mục này. Khi tích hợp từng phần, cập nhật tại README này tên notebook/script, thứ tự chạy, phiên bản thư viện và vị trí input/output. Nếu phần nào cần hướng dẫn dài, có thể bổ sung README riêng khi cần.

## Honduras data validation

Notebook:

- `data/validation/honduras_data_validation.ipynb`

### Mục tiêu

Xác minh nhãn và chất lượng dữ liệu Honduras, giữ nguyên account-level
train/validation/test split và tạo các feature cấp tài khoản phục vụ E2/E3.

### Input

Notebook sử dụng các file đã preprocessing:

- `honduras_bad_normalized.jsonl`
- `honduras_good_normalized.jsonl`
- `honduras_account_manifest.csv`
- `honduras_account_splits.csv`
- các file audit duplicate/conflict trong `quality_audit/`

Dữ liệu JSONL gốc/normalized có kích thước lớn và được lưu ngoài Git.

### Output

- `data/validation/honduras_features.parquet`
- `data/validation/data_quality_summary.csv`
- `data/validation/honduras_feature_profile.csv`

Feature table gồm:

- `account_id`
- `label`
- `split`
- `tweet_count`
- `avg_text_length`
- `url_ratio`
- `hashtag_ratio`
- `mention_ratio`
- `retweet_ratio`

### Kết quả validation

- Tổng số account: 224,685
- Bad (`label=0`): 1,866
- Good (`label=1`): 222,819
- Train: 157,279
- Validation: 22,468
- Test: 44,938
- Account overlap giữa các split: 0
- Final label mismatch: 0
- Final split mismatch: 0
- Missing/extra account so với manifest: 0

### Thứ tự chạy

1. Mount Google Drive trong Colab.
2. Cập nhật `BASE_DIR`/`OUTPUT_DIR` trong notebook theo vị trí dữ liệu.
3. Chạy notebook từ đầu đến cuối (`Runtime -> Run all`).
4. Notebook kiểm tra label, missing, duplicate và split.
5. Aggregate tweet thành account-level feature table.
6. Chạy final QA và export các output.

Lưu ý: Honduras có class imbalance lớn, Bad chỉ chiếm khoảng 0.83%.
Các experiment downstream phải giữ nguyên split hiện tại và không nên dùng
Accuracy làm metric chính.

Dữ liệu gốc và model lớn lưu ngoài repo; ghi nguồn tải và đường dẫn đặt file trong hướng dẫn chạy. Giữ split tài khoản đã thống nhất, không tự chia lại trong từng experiment.

Kết quả hiện có là baseline, chưa chứng minh hiệu quả transfer learning. Hướng dẫn kiểm tra hiện tại và giới hạn nằm trong [evaluation/README.md](evaluation/README.md); các đường dẫn local trong script còn cần cập nhật khi tích hợp.

Quy ước nhánh, commit và Pull Request: [CONTRIBUTING.md](CONTRIBUTING.md). Mỗi phần làm trên nhánh riêng rồi tạo PR vào `main`.
