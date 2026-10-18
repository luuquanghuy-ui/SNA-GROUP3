# Báo cáo đánh giá và tái lập baseline

Cập nhật 18/10/2026. Các kiểm tra dưới đây dùng file local của Người 0–4 trong `D:\mxhui`; không tạo lại số liệu thực nghiệm và không đẩy dữ liệu lên GitHub.

## Kết luận theo từng phần

| Phần | Đã kiểm tra độc lập | Chưa kiểm tra |
| --- | --- | --- |
| Người 1 — dữ liệu Honduras | Chạy lại các cell xử lý của notebook theo thứ tự trên processed input Người 0, chỉ thay đường dẫn Colab/Drive bằng đường dẫn local. `honduras_features.csv`, `data_quality_summary.csv` và `honduras_feature_profile.csv` khớp **về toàn bộ giá trị** với bản bàn giao. Feature table khớp manifest split theo account ID/label/split. | Chưa chạy lại bước Người 0 tạo normalized JSONL từ hai file Honduras gốc. |
| E1 — Fox8 Bot/Human | Tài khoản/nhãn test khớp split Người 0. Đã suy luận lại từ NDJSON + checkpoint và **huấn luyện lại** 3 model bằng các cell huấn luyện notebook, chỉ thay bước đọc ZIP bằng NDJSON local. Bot-F1 của cả 3 model giống bản bàn giao; nhãn dự đoán test khớp **100%**. | Chưa chạy notebook nguyên bản không sửa đường dẫn trên Colab; xác suất có sai khác nhỏ do phiên bản thư viện. |
| E2 — Honduras Bad/Good | Test prediction khớp đúng 44.938 tài khoản và nhãn Người 1. Huấn luyện lại DummyClassifier và LogisticRegression từ bảng feature bàn giao; nhãn dự đoán trên validation/test khớp **100%**, chênh xác suất tối đa dưới `6×10⁻¹³`. Chỉ số tính lại khớp `e2_metrics.csv`. | Chưa chạy notebook nguyên bản trên Colab; lần chạy lại dùng script độc lập và scikit-learn phiên bản khác. |
| E3 — graph hashtag–giờ | Đã chạy lại cell 2–4 của notebook từ JSONL Người 0, chỉ đổi đường dẫn và cách hiển thị. Bảng chất lượng, edge list, event audit và components khớp từng dòng sau khi chuẩn hóa ký tự xuống dòng. Cả 224.685 node khớp ID, nhãn, split, degree và strength. | Louvain cho phân hoạch khác: 2.092 cộng đồng ở lần chạy local so với 2.101 trong bản bàn giao. Máy local dùng NetworkX 3.6.1, notebook gốc ghi 3.7; cùng 3.6.1 nhưng đổi `PYTHONHASHSEED` cũng đổi phân hoạch. Không gọi kết quả cộng đồng là tái lập chính xác. |

## Sai khác và cách xử lý

1. **CSV Người 1 khác SHA-256 sau khi chạy lại.** Nội dung bảng đặc trưng khớp chính xác theo account ID ở mọi cột; quality summary và feature profile cũng khớp theo giá trị. Vì vậy ghi “tái lập về nội dung”, không ghi “file byte-for-byte giống nhau”. Chi tiết ở `honduras_preprocessing_rerun_audit.json`.
2. **E1/E2 được lưu bằng scikit-learn 1.6.1, môi trường kiểm tra dùng 1.8.0.** Phiên bản gốc được xác định từ cảnh báo khi nạp các estimator đã lưu; E1 vẫn khớp 100% nhãn, E2 chạy lại từ đầu cũng khớp 100% nhãn. Cảnh báo phiên bản vẫn cần ghi trong báo cáo, nhất là khi nhóm muốn chạy lại trong môi trường mới.
   Lần huấn luyện lại E1 cho Bot-F1 bằng bản bàn giao ở cả ba model. Chênh xác suất lớn nhất là khoảng `6,53×10⁻⁹` (Logistic Regression), `0,01` (Random Forest) và `1,65×10⁻⁶` (MLP); các nhãn tại ngưỡng hiện dùng vẫn giống nhau.
3. **Thông báo E3 trong nhóm khác bộ file bàn giao.** Tin nhắn ghi 2.098 cộng đồng/modularity khoảng 0,6544; notebook, `e3_graph_metrics.csv`, `E3_RESULTS.md` và phép tính lại độc lập đều cho **2.101 cộng đồng, modularity 0,654076266**. Đây là số dùng khi báo cáo bộ file hiện có; chỉ cập nhật nếu Người 4 bàn giao một bản chạy mới.
4. **`e2_config.json` gọi project là “Honduras Bot Detection”.** Đây là nhãn đặt tên chưa đúng. Tác vụ E2 thực tế là phân loại tài khoản Bad/Good; Bad không mặc định là bot.
5. **Chạy lại E3 từ tweet xác nhận graph nhưng chưa tái lập đúng Louvain.** Lần chạy local cho 2.092 cộng đồng, cộng đồng lớn nhất 17.153 node và modularity 0,655466348; bản bàn giao lần lượt là 2.101, 17.482 và 0,654076266. Edge list, event audit, quality summary và components trùng nội dung với bản bàn giao; khác biệt byte chỉ do xuống dòng Windows/Colab. Với NetworkX 3.6.1 trên cùng graph, `PYTHONHASHSEED=0` cho 2.093 cộng đồng/modularity 0,654404007; `PYTHONHASHSEED=1` cho 2.102/0,655142759. Như vậy `seed=42` trong Louvain chưa đủ cố định kết quả trong quy trình này. Chưa chạy lại dưới đúng NetworkX 3.7 của Nam nên không quy nguyên nhân chênh lệch duy nhất cho phiên bản thư viện.

## Giới hạn học thuật không được bỏ qua

- E1 Bot-F1 gần 1,0 trên Fox8-23 chưa chứng minh detector tổng quát, vì Bot và Human gắn với các nguồn thu thập khác nhau.
- E2 hiện dùng sáu feature thống kê, có `tweet_count`, trong khi E1 dùng TF-IDF văn bản. Vì thế **checkpoint E1 chưa có đầu vào tương thích để chuyển trực tiếp sang E2**. Chưa có phép so scratch với Fox8-pretrained trong dữ liệu bàn giao.
- LogisticRegression E2 có Bad-Recall 0,965 nhưng Bad-Precision 0,089: gắn nhầm 3.679 Good là Bad. Không gọi đây là “phát hiện bot tốt” hoặc chỉ nêu Recall mà bỏ Precision.
- E3 là graph hashtag–giờ trên **toàn corpus**, không phải graph trên cùng test E2 trong proposal. Chỉ có một cấu hình graph; chưa có đối chứng embedding transfer, ablation hoặc đánh giá campaign theo nhãn chuẩn.
- Mức độ tài khoản Bad có cạnh (1.100/1.866) cao hơn Good (83.719/222.819), nhưng mức hoạt động của hai lớp cũng khác nhiều. Độ phủ này không chứng minh các cộng đồng là chiến dịch phối hợp độc hại.

## File bằng chứng

| File | Nội dung |
| --- | --- |
| `honduras_features_audit.json` | Schema, ID, nhãn, split, feature values và checksum đầu vào Người 1. |
| `honduras_preprocessing_rerun_audit.json` | Các cell đã chạy, đối chiếu từng output Người 1 theo nội dung và checksum. |
| `e1_inference_audit.json` | Kiểm tra dự đoán E1 từ Fox8 NDJSON gốc và checkpoint. |
| `e1_training_rerun_audit.json`, `e1_training_rerun_log.txt` | Huấn luyện lại E1 bằng các cell notebook và so với output gốc. |
| `e2_local_rerun_audit.json` | Huấn luyện lại E2, đối chiếu nhãn và xác suất. |
| `e3_graph_recalculation.json` | Tính lại graph metrics từ node/edge. |
| `baseline_handoff_audit.json` | Đối chiếu E1/E2 prediction, E3 output và SHA-256 input/model. |
| `baseline_results.csv` | 67 dòng chỉ số baseline đã xác minh; nguồn mỗi dòng ở cột `input`. |
| `e3_rerun_from_tweets/comparison.json` | Đối chiếu lần dựng graph từ tweet với bản Nam: bảng nào trùng, trường node nào trùng và ba chỉ số Louvain nào lệch. |

Các phép so transfer và đánh giá trên test-only trong proposal là **việc nghiên cứu tiếp theo của nhóm**; không được tự ghi kết quả vào bảng baseline khi chưa chạy.
