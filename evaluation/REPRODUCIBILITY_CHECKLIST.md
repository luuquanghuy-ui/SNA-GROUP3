# Checklist đánh giá và tái lập kết quả

Cập nhật: 18/10/2026. Chỉ đánh dấu **xong** khi có tệp đầu ra và đã kiểm tra được nội dung. Checklist này theo dõi phần đánh giá baseline; việc tổ chức repository được theo dõi riêng.

## 1. Chốt quy ước chung

- [x] Đọc phân công mới và proposal; phân biệt ba nhiệm vụ E1, E2, E3.
- [x] Xác định nhãn dương: E1 là Bot theo mã nhãn do người làm E1 xác nhận; E2 là Bad (`label=0`), Good là `label=1`.
- [x] Lập quy ước ghi kết quả theo `experiment`, `run_id`, `model`, `split`, `seed`, `metric`, `value`, `input`, `feature_set`, `notes`.
- [x] Tạo mẫu manifest cho từng lần chạy để ghi checksum, tham số, môi trường và đường dẫn output.
- [x] Nhận E1: Bot=1, seed 42, TF-IDF văn bản, prediction/metrics, vectorizer và checkpoint.
- [x] Nhận E2: Bad=0, seed 42 trong notebook, sáu feature thống kê, prediction/metrics, scaler/model; xác nhận test đủ 44.938 tài khoản.
- [x] Nhận E3: graph hashtag–giờ, Louvain seed 42, node/edge/community, metrics và config.
- [x] Tìm thấy manifest/split Fox8 và toàn bộ processed Honduras trong `nguoi0`; kiểm tra khớp test E1 và SHA-256 input E3.
- [ ] Ghi đủ phiên bản môi trường huấn luyện gốc E1/E2 ngoài scikit-learn 1.6.1; notebook E1 còn tham chiếu ZIP trong khi dữ liệu local là NDJSON.

## 2. Kiểm tra dữ liệu đầu vào đã bàn giao

- [x] Kiểm tra trực tiếp `nguoi 1/honduras_features.csv` và `data_quality_summary.csv`; lưu kết quả kiểm tra ở `honduras_features_audit.json`.
- [x] Đối chiếu số tài khoản, nhãn, split, giá trị thiếu, ID trùng và miền giá trị của các tỷ lệ.
- [x] Chạy lại toàn bộ cell xử lý của notebook Người 1 trên input processed `nguoi0` sau khi thay đường dẫn Colab bằng local; ba CSV output khớp theo giá trị, checksum byte khác.
- [x] Ghi checksum các prediction, metrics, graph node/edge và model/checkpoint đã bàn giao trong `baseline_handoff_audit.json`.
- [x] Đối chiếu cả bảy SHA-256 input E3 trong config với bản processed `nguoi0`; tất cả khớp.

## 3. Tổng hợp kết quả E1/E2/E3

- [x] Tạo `baseline_results.csv` với 67 dòng metric E1/E2/E3 được tính hoặc kiểm tra từ output thật.
- [x] Tạo `BASELINE_RESULTS_SUMMARY.md` ghi rõ câu hỏi, đơn vị đánh giá, nhãn và trạng thái từng thí nghiệm.
- [x] Tính lại E1 từ prediction và đối chiếu Bot-F1, Precision, Recall, Macro-F1, AP, confusion matrix với e1_metrics.csv.
- [x] Tính lại E2 từ prediction; đối chiếu Bad-F1, Precision, Recall, Macro-F1, AP và tập test với Người 1.
- [x] Kiểm tra E3 node/edge/community, graph metrics và độ phủ node có cạnh theo nhãn Bad/Good.
- [x] Xác nhận các model trong từng E1/E2 dùng cùng test; E3 chỉ có một baseline trên full corpus, chưa có cấu hình để so.
- [ ] Có E2 scratch/transfer dùng cùng input và E3 baseline/enhanced dùng cùng tập test để thực hiện các phép so trong proposal.
- [ ] Ghi nhận các lần chạy thất bại hoặc kết quả âm tính, không loại khỏi báo cáo chỉ vì điểm thấp.

## 4. Tái lập và báo cáo

- [x] Viết `README.md` chỉ đường dẫn input/output, thứ tự chạy và giới hạn hiện tại.
- [x] Viết `EVALUATION_PROTOCOL.md` để nhóm bàn giao tệp đúng schema.
- [x] Huấn luyện lại E2 từ bảng feature bằng script độc lập; cả hai model khớp 100% nhãn dự đoán validation/test.
- [x] Suy luận lại E1 từ Fox8 NDJSON, vectorizer/checkpoint; cả ba model khớp 100% nhãn test.
- [x] Huấn luyện lại ba model E1 bằng các cell notebook trên Fox8 NDJSON local; Bot-F1 và nhãn test khớp bản bàn giao.
- [x] Tính lại độc lập E3 từ edge/node: density, degree, strength, components, cộng đồng và modularity đều khớp.
- [x] Dựng lại graph E3 từ tweet và chạy Louvain trên runtime local; cạnh và thống kê cấu trúc khớp, số cộng đồng khác bản bàn giao. Đã lưu `e3_rerun_from_tweets/comparison.json`.
- [x] Ghi runtime của lần chạy lại E2, seed/config tìm thấy trong notebook và checksum checkpoint đã bàn giao.
- [x] Lưu phiên bản môi trường local trong `requirements_evaluation.txt`; xác định model E1/E2 đã lưu bằng scikit-learn 1.6.1 qua cảnh báo khi nạp.
- [ ] Bổ sung các phiên bản gốc còn thiếu và tái lập đúng phân hoạch Louvain E3 dưới NetworkX 3.7 với môi trường/hash seed được ghi rõ.
- [x] Cập nhật README và bảng tổng hợp theo các file bàn giao hiện có.
- [x] Mỗi số baseline đã báo có đường dẫn đến file nguồn và kiểm tra trong `baseline_handoff_audit.json`.

## 5. Những điểm cần báo rõ với nhóm

- Bảng đặc trưng Người 1 là đầu vào Honduras E2; nó không tự tạo checkpoint transfer từ Fox8 và không đủ để tái lập E1/E3.
- Notebook Người 1 hiện dùng đường dẫn Google Drive tuyệt đối; chưa thể gọi là chạy lại độc lập trong thư mục này.
- Baseline E2 trên các feature thống kê là mốc phân loại Bad/Good; chưa phải thí nghiệm chuyển giao từ Fox8.
- Baseline graph E3 là hashtag–giờ trên toàn corpus, không phải test-only theo proposal; không được báo F1 phát hiện chiến dịch khi thiếu nhãn chiến dịch.
- Tin nhắn nhóm ghi 2.098 community/0,6544 modularity; notebook, file metrics, `E3_RESULTS.md` và phép tính lại từ graph đều cho 2.101/0,654076. Dùng số của bộ file bàn giao hiện tại; chỉ đổi khi có bản chạy mới.
- E1 dùng TF-IDF văn bản còn E2 baseline dùng sáu feature thống kê; chưa thể gọi kết quả hiện tại là transfer từ Fox8.
- Không dùng `retweet_tweetid` đã bị làm tròn để khẳng định hai tài khoản retweet đúng cùng một bài nếu chưa xác minh ID.
