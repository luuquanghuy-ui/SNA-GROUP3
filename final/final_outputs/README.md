# Bàn giao Thành viên 5

Kiểm định trên nhánh final tại commit ccc53ac1b5339832c3676a9e0f2477b432dded5f.

- [Bảng E2 bàn giao](tables/e2_comparison.csv), [E2 train lại theo seed](audit/e2_retrained_test_by_seed.csv) và [đối chiếu hai checkpoint E1](audit/e2_e1_checkpoint_comparison.csv).
- [Bảng E3](tables/e3_comparison.csv) phân biệt Graph A, Graph B bàn giao, Graph B lọc tweet và Graph B lọc tweet dùng checkpoint E1 huấn luyện lại.
- [Metrics E1 train lại](audit/e1_retrained_test_metrics.json), [kiểm tra đặc trưng](audit/feature_eda.json) và [audit leakage](audit/leakage_sensitivity.json).

E1 đã huấn luyện lại từ dữ liệu được cung cấp. E2 đã chạy lại chín checkpoint bàn giao và huấn luyện lại ba cấu hình qua ba seed với checkpoint E1 mới. E3 đã tạo lại vector, dựng cạnh và chạy Leiden. Graph A bàn giao có 0 cạnh tương tác xác thực, nên ARI A–B và đối chứng cùng số cạnh không áp dụng. Audit đo độ nhạy trước các ID trùng giữa split gốc E1 và Test E2. Source confounding của Honduras và việc xác nhận campaign cần thêm bằng chứng.

Thư mục audit chỉ chứa kết quả kiểm định tổng hợp. Dữ liệu thô, cache, checkpoint, dự đoán theo ID và ảnh kiểm bố cục được giữ trong workspace.

## Đối chiếu phiên bản trước khi bàn giao

Ngày 25/11/2026, fetch từ GitHub xác nhận nhánh final vẫn ở ccc53ac1b5339832c3676a9e0f2477b432dded5f. Checkout kiểm định và checkout chuẩn bị bàn giao cùng commit. SHA-256 của embedding Honduras đã dùng trùng OID Git LFS trên nhánh final. Các script Python E1 trong thư mục huấn luyện lại khớp SHA-256 với bản nhóm.

E1 chạy lại cùng script và ba split gốc cùng hai file hàng xóm do nhóm đưa. E2 chạy các cell huấn luyện và đánh giá của notebook nhóm trên CPU, đổi đường dẫn Colab sang workspace. Có hai lượt riêng với checkpoint E1 bàn giao và checkpoint E1 huấn luyện lại. E3 lọc tweet là phân tích độ nhạy của Thành viên 5. Nó dùng cùng cosine, top-k 5, ngưỡng 0,90 và Leiden, nhưng loại đầu vào thiếu tweet và tìm top-k bằng PyTorch thay FAISS. Kết quả đã tách khỏi Graph B bàn giao.
