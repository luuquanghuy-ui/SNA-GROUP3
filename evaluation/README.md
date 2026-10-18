# Evaluation and reproducibility

Thư mục này chứa mã kiểm tra dữ liệu, đánh giá ba baseline E1/E2/E3 và đối chiếu khả năng tái lập. Các số liệu là kết quả đã bàn giao ở thời điểm hiện tại, không phải kết quả cuối của đồ án.

## Nội dung

| Nhóm file | Vai trò |
| --- | --- |
| `EVALUATION_PROTOCOL.md`, `run_manifest_template.json` | Quy ước nhãn, split, seed, metric và metadata cho mỗi lần chạy |
| `aggregate_baseline_results.py`, `evaluate_predictions.py` | Tính lại metric từ prediction/graph và lập `baseline_results.csv` |
| `audit_honduras_features.py`, `rerun_honduras_preprocessing.py` | Kiểm tra bảng đặc trưng Honduras và chạy lại bước tạo bảng |
| `verify_e1_inference.py`, `retrain_e1.py`, `reproduce_e2.py` | Suy luận và huấn luyện lại baseline E1/E2 |
| `verify_e3_graph.py`, `rerun_e3_from_tweets.py`, `audit_e3_full_rerun.py` | Kiểm tra graph E3 và dựng lại từ tweet |
| `probe_e3_louvain_stability.py` | Kiểm tra độ ổn định của phân hoạch Louvain khi thay Python hash seed |
| `BASELINE_RESULTS_SUMMARY.md`, `REPRODUCIBILITY_REPORT.md`, `REPRODUCIBILITY_CHECKLIST.md` | Tóm tắt kết quả, bằng chứng và giới hạn kiểm tra |
| `*_audit.json`, `e3_rerun_from_tweets/*.json` | Đầu ra kiểm tra; không chứa toàn bộ graph hoặc dữ liệu gốc |

## Phạm vi tái lập hiện tại

- Bảng đặc trưng Honduras, nhãn/split và dự đoán E1/E2 đã được đối chiếu với các file bàn giao; E1/E2 đã được chạy lại trên máy kiểm tra.
- E3 đã được dựng lại từ tweet đã chuẩn hóa. Edge list, event audit, components và các thuộc tính cấu trúc của node khớp bản bàn giao sau khi chuẩn hóa kiểu xuống dòng.
- **Lần chạy lại tái lập graph và các thống kê cấu trúc, nhưng chưa tái lập đúng phân hoạch Louvain của bản bàn giao.** Các chỉ số cộng đồng trong `baseline_results.csv` là số của bản bàn giao, không phải số cộng đồng đã tái lập khớp.
- E2 hiện phân loại Bad/Good bằng sáu đặc trưng thống kê; chưa có kết quả transfer từ encoder Fox8. E3 hiện chỉ có graph baseline.

## Đầu vào và môi trường

Mã được viết và kiểm tra với cấu trúc bàn giao **trên máy local**: các thư mục nguồn dữ liệu và output E1/E2/E3 nằm cùng cấp với thư mục `evaluation`. Đường dẫn `nguoi0`, `nguoi 1`, `nguoi2`, `nguoi3`, `nguoi4` trong code là **tên folder local để nhận file**, không phải tên thành viên hay cấu trúc cuối của repo. Dữ liệu gốc, graph lớn và model được lưu ngoài thư mục này.

Sau khi các phần dữ liệu/E1/E2/E3 được merge vào repo, cần cập nhật các hằng đường dẫn đầu vào trong script theo vị trí file thực tế và chạy kiểm tra lại. Hiện chưa thể bảo đảm clone riêng thư mục này rồi chạy được trên máy khác. Phiên bản thư viện của lần kiểm tra local nằm trong `requirements_evaluation.txt`; notebook gốc có thể dùng phiên bản khác.

Khi có đủ dữ liệu và đường dẫn đã được cập nhật, chạy các lệnh từ thư mục `evaluation` theo thứ tự:

```powershell
python -B .\audit_honduras_features.py
python -B .\rerun_honduras_preprocessing.py
python -B .\aggregate_baseline_results.py
python -B .\verify_e1_inference.py
python -B .\retrain_e1.py
python -B .\reproduce_e2.py
python -B .\verify_e3_graph.py
python -B .\rerun_e3_from_tweets.py
python -B .\audit_e3_full_rerun.py
```

`rerun_e3_from_tweets.py` dùng nhiều RAM vì dựng graph từ toàn bộ Honduras. Đầu ra lớn được tạo lại tại `e3_rerun_from_tweets/`; repo chỉ giữ các file audit nhỏ để tránh đưa graph lặp lên GitHub.
