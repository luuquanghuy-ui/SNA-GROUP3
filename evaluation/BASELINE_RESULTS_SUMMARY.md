# Tổng hợp kết quả đã kiểm tra — E1/E2/E3

Cập nhật 18/10/2026. Chỉ số E1/E2 bên dưới được tính lại từ các file prediction bàn giao và khớp với metrics CSV của từng người. Chỉ số E3 khớp bảng node/edge/community và metrics CSV. Bảng đầy đủ từng metric nằm ở `baseline_results.csv`; checksum và các phép đối chiếu ở `baseline_handoff_audit.json`.

## E1 — Bot/Human trên Fox8-23

Lớp dương Bot = `1`. Cả ba model dùng cùng **456 tài khoản test** (228 Bot, 228 Human).

| Model | Bot-Precision | Bot-Recall | Bot-F1 | Macro-F1 | AP | TP/FP/TN/FN |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 228/0/228/0 |
| Random Forest | 0,9956 | 1,0000 | 0,9978 | 0,9978 | 1,0000 | 228/1/227/0 |
| MLP | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 228/0/228/0 |

Nguồn: `../nguoi2/evaluation-20261018T031128Z-1-001/evaluation/e1_predictions.csv`. E1 lấy văn bản tối đa 30 tweet/tài khoản, tạo TF-IDF rồi huấn luyện model. `dataset` không được đưa làm feature trực tiếp, nhưng Bot thuộc Fox8 còn Human thuộc bốn nguồn khác; điểm gần hoàn hảo có thể phản ánh khác biệt nguồn/văn bản. **Không kết luận mô hình phát hiện bot tổng quát.** Đã đối chiếu test với split trong `nguoi0`, suy luận lại từ checkpoint và huấn luyện lại ba model từ NDJSON; Bot-F1 và nhãn test khớp bản bàn giao.

## E2 — Bad/Good trên Honduras

Lớp dương Bad = `0`. Cả hai model dùng đúng **44.938 tài khoản test** của Người 1: 373 Bad và 44.565 Good; không undersampling test.

| Model | Bad-Precision | Bad-Recall | Bad-F1 | Macro-F1 | AP | TP/FP/TN/FN |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dummy Classifier | 0 | 0 | 0 | 0,4979 | 0,0083 | 0/0/44565/373 |
| Logistic Regression | 0,0891 | 0,9651 | 0,1632 | 0,5600 | 0,4712 | 360/3679/40886/13 |

Nguồn: `../nguoi3/evaluation-20261018T031434Z-1-001/evaluation/e2_predictions.csv`. Baseline dùng sáu feature thống kê cấp tài khoản, gồm `tweet_count`; do đó cần kiểm tra riêng ảnh hưởng của mức hoạt động trước khi diễn giải khả năng học “hành vi”. Logistic Regression tìm được 360/373 tài khoản Bad nhưng cũng gắn nhầm 3.679 Good. Đây là **Bad/Good classification**, chưa phải transfer từ Fox8 hoặc phát hiện bot trên Honduras.

Đã huấn luyện lại độc lập từ `../nguoi 1/honduras_features.csv` theo đúng StandardScaler, DummyClassifier và LogisticRegression trong notebook E2. Nhãn dự đoán validation/test khớp **100%** với file bàn giao; chênh xác suất lớn nhất dưới `6×10⁻¹³`. Kết quả và phiên bản môi trường nằm trong `e2_local_rerun_audit.json`.

## E3 — graph hashtag–giờ trên toàn Honduras

Graph nối hai tài khoản cùng dùng một hashtag trong cùng giờ UTC; bỏ event trên 100 tài khoản, giữ cạnh có weight ≥ 2. Louvain seed 42 được chạy trên các đỉnh có cạnh; isolate mang community `-1`.

| Chỉ số | Giá trị |
| --- | ---: |
| Tổng node | 224.685 |
| Node có cạnh | 84.819 |
| Node cô lập | 139.866 |
| Cạnh | 1.905.386 |
| Thành phần liên thông | 141.752 |
| Cộng đồng trên node có cạnh | **2.101** |
| Modularity có trọng số | **0,654076** |
| Cộng đồng lớn nhất | 17.482 node |

Trong graph có 1.100/1.866 tài khoản Bad có cạnh (58,95%); 83.719/222.819 Good có cạnh (37,57%). Đây là **độ phủ của graph theo nhãn tài khoản**, không phải recall phát hiện chiến dịch. Chênh lệch hoạt động giữa Bad và Good có thể ảnh hưởng khả năng xuất hiện cạnh. Dữ liệu dùng **toàn corpus**, nên không so trực tiếp kết quả này như đánh giá E3 trên tập test E2. Không có ground truth campaign để tính Precision/Recall/F1 campaign.

Nguồn: `../nguoi4/E3_outputs-20261018T031701Z-1-001/E3_outputs/`. **Kết quả cuối trong bộ file bàn giao là 2.101 cộng đồng và modularity 0,654076**: `e3_graph_metrics.csv`, `E3_RESULTS.md` và output trong `E3_Coordination_Graph.ipynb` đều khớp; phép tính lại từ bảng node/edge cũng khớp. Tin nhắn nhóm từng ghi 2.098/0,6544 nhưng không khớp bộ file này. Khi báo cáo bộ file hiện có, dùng 2.101/0,654076.

SHA-256 của bảy input E3 khớp bản processed trong `nguoi0`. Tính lại components, degree, strength, density và weighted modularity từ bảng node/edge bản bàn giao cũng khớp file metrics. Đã chạy lại bước dựng graph từ tweet: edge list, event audit, quality summary, components và mọi trường node ngoài `community` khớp bản bàn giao. **Lần chạy lại tái lập graph và các thống kê cấu trúc, nhưng chưa tái lập đúng phân hoạch Louvain của bản bàn giao.** Louvain local cho 2.092 cộng đồng/modularity 0,655466, khác bản Nam 2.101/0,654076; thay `PYTHONHASHSEED` trong cùng NetworkX 3.6.1 cũng làm đổi cộng đồng. Vì vậy các số cộng đồng ở bảng trên là **kết quả bản Nam bàn giao**, chưa được tái lập chính xác trên runtime local. Chi tiết ở `e3_rerun_from_tweets/comparison.json`.

## Những phép so còn thiếu so với proposal

| Câu hỏi trong proposal | Trạng thái hiện tại |
| --- | --- |
| E2 scratch so với Fox8-pretrained + fine-tuning | Chưa làm. E1 dùng TF-IDF văn bản, E2 baseline dùng sáu feature thống kê; chưa có đầu vào chung để nạp checkpoint E1 vào E2. |
| E3 baseline graph so với graph có embedding scratch/transferred | Chưa làm. Hiện chỉ có một graph hashtag–giờ, chưa có cấu hình bổ sung embedding. |
| E3 đánh giá trên cùng test Honduras và cùng tập đỉnh | Chưa làm; graph hiện dùng toàn corpus. |
| Ablation/độ nhạy/số cạnh bằng nhau | Chưa có kết quả. |
| Chạy lại E1/E3 từ dữ liệu gốc trên runtime mới | Đã huấn luyện lại E1 và dựng lại graph E3 từ tweet. Cạnh và các thống kê cấu trúc E3 khớp; Louvain khác do quy trình còn nhạy với thứ tự duyệt/phiên bản runtime. Chưa tạo virtual environment riêng hoặc tái lập đúng NetworkX 3.7. |

Kết luận hiện tại chỉ giới hạn ở **ba baseline đã chạy**. Không viết rằng transfer learning đã cải thiện E2/E3 hoặc rằng cộng đồng graph là campaign thật.
