# Giới hạn và khả năng tái lập

## Phạm vi dữ liệu

Test E2 có 7.728 trong tổng số 44.938 tài khoản Test gốc. Tập bị loại gồm 28 Bad và 37.182 Good. Phân bố nhãn sau lọc khác toàn bộ Test. Khả năng tổng quát hóa cho tài khoản thiếu tweet chưa được đo.

## Leakage và nguồn dữ liệu

Các ID trong metadata đã đối chiếu với node E3. Code E2 loại nội dung trùng chính xác sau chuẩn hóa khoảng trắng giữa các split. Đây là kiểm tra hữu hạn, còn kiểm tra trùng gần đúng, nguồn gán nhãn, và trùng giữa nguồn E1 và Honduras cần dữ liệu gốc và provenance đủ hơn. Số đo E1 mới không xác định được nguyên nhân kết quả bản cũ.

## Diễn giải đồ thị

Graph A chưa đủ dữ liệu tương tác để so A–B. ARI và đối chứng cùng số cạnh không áp dụng. Nhãn Bad/Good là nhãn account. Chưa có ground truth ở cấp campaign để đo chất lượng phát hiện campaign. Weighted modularity và độ phủ Bad chỉ mô tả graph và cách chia nhóm.

## Bằng chứng

Script audit/replay_e2_e3.py đã tạo dự đoán theo seed và bản E3 có lọc tweet sau khi tải ma trận embedding qua Git LFS. Phần kiểm định dùng artifact đã lưu và nhánh final tại ccc53ac.

## Kiểm tra nguồn bổ sung

SHA-256 của cả chín file nguồn Honduras khớp cấu hình Người 1. So ID Test TwiBot-20 với Honduras cho thấy 11 ID chung. Trong đó, E2 Train có 4 ID (2 có tweet), Validation có 2 (cả 2 có tweet), Test có 5 (3 có tweet và được chấm). E1 đã dùng Dev để chọn checkpoint và Test để đánh giá. Các ID chung này xác nhận hai nguồn có overlap, còn kết luận leakage vào E2 Test cần đối chiếu riêng ID từ Train và Dev của E1, cũng như provenance và nội dung. Chi tiết nằm trong audit/source_validation.json.

## Đối chiếu ID từ split gốc TwiBot-20

Đã đọc train.json và dev.json trong thư mục task. Có 8.278 ID Train và 2.365 ID Dev, không trùng ID giữa hai split. Trong 7.728 tài khoản Test E2 được chấm, 11 ID trùng Train E1 gốc và 5 ID trùng Dev E1 gốc. Cả 16 tài khoản đều mang nhãn Good ở Honduras. Nếu E1 giữ nguyên danh sách tài khoản gốc, đây là overlap với dữ liệu huấn luyện và chọn checkpoint E1. Cache thực tế vẫn là bằng chứng cần để xác nhận membership của lần chạy. Kết quả chi tiết nằm trong audit/e1_raw_split_overlap.json.

## Kiểm tra độ nhạy khi loại tài khoản trùng E1

File Test TwiBot-20 gốc có đúng 1.183 ID, trùng hoàn toàn với tập ID trong dự đoán Test E1. Cả ba split gốc Train, Dev và Test không trùng ID từng cặp. Trong Test E2 chính, có 11 ID trùng Train E1, 5 ID trùng Dev E1 và 3 ID trùng Test E1. Cả 19 mang nhãn Good ở E2. Sau khi loại 16 ID trùng Train hoặc Dev, tập đánh giá còn 7.712 tài khoản. Tính lại từ dự đoán của chín checkpoint, mức chênh tuyệt đối lớn nhất là 0,0007284 Macro-F1, 0,0013957 Bad F1 và 0,0004878 Bad AP. Đây là phép đo độ nhạy trên checkpoint đã lưu. Cache Train/Dev của lượt chạy E1 vẫn chưa được xác minh. Bằng chứng nằm trong audit/leakage_sensitivity.json.

## Kiểm tra đường đi của nhãn và nguồn hàng xóm

Trong code E1, nhãn được lưu riêng để huấn luyện và đánh giá. Các nhánh nhận nội dung tweet, văn bản hồ sơ, các trường số và Boolean của hồ sơ, và văn bản hồ sơ hàng xóm. Bộ chuẩn hóa đặc trưng số fit trên Train. E2 dùng mảng MiniLM làm đầu vào và mảng nhãn riêng làm mục tiêu. E3 tạo cạnh từ biểu diễn và dùng Bad/Good để thống kê sau phân cụm. Rà code chưa thấy nhãn được đưa trực tiếp vào vector đầu vào hay điều kiện tạo cạnh. Kết quả này giới hạn ở luồng code đã kiểm, chưa xác nhận dữ liệu thô không chứa tín hiệu gián tiếp gắn với nguồn hoặc nhãn.

Đã đối chiếu hồ sơ hàng xóm được chọn và có bản ghi support với danh sách tài khoản gán nhãn của ba split E1. Không thấy ID trùng trong chín cặp split. Audit này kiểm nguồn hồ sơ hàng xóm theo ID. Source confounding của Honduras vẫn chưa thể loại trừ vì cần thông tin thu thập và nguồn gán nhãn đủ chi tiết ở cấp tài khoản. Tỷ lệ thiếu tweet khác rõ giữa Bad và Good, và phép kiểm toàn bộ Test E2 đã định lượng ảnh hưởng của nhóm đầu vào zero. Xem audit/neighbor_provenance.json và audit/e2_cohort_sensitivity.csv.
