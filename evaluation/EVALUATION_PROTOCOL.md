# Quy ước bàn giao và so sánh kết quả

## Dữ liệu, nhãn và split

| Mục | Đơn vị | Nhãn dương | Split đánh giá | Ý nghĩa |
| --- | --- | --- | --- | --- |
| E1 | Tài khoản Fox8-23 | Bot, mã số phải xác nhận từ file E1 | Test theo account ID | Phân loại Bot/Human trên dữ liệu nguồn |
| E2 | Tài khoản Honduras | Bad = `0`; Good = `1` | Test cố định của Người 1, không undersampling | Phân loại Bad/Good, không phải bot |
| E3 | Mạng tài khoản Honduras | Không có nhãn dương campaign | Cùng tập đỉnh test Honduras cho mọi graph được so | Cấu trúc phối hợp và mức tập trung Bad trong cộng đồng |

Người làm E1/E2 phải gửi file prediction cấp tài khoản gồm `account_id,split,y_true,y_pred,y_score_positive`, trong đó điểm liên tục là điểm của **lớp dương**. Nếu không có điểm, bỏ cột này; AP không được ghi. Kèm tên model, feature schema, preprocessing, ngưỡng dự đoán, seed, kích thước train/validation/test, phiên bản thư viện, checkpoint và hash input. Không đưa `account_id`, `split`, nhãn hay tên nguồn vào tập feature dự đoán.

## Chỉ số phân loại

E1 dùng Bot-F1 làm chỉ số chính; báo Precision, Recall, Macro-F1, AP nếu có score, và TP/FP/TN/FN. E2 dùng Bad-F1 làm chỉ số chính; báo Precision, Recall, Macro-F1, AP và TP/FP/TN/FN. AP được tính từ điểm liên tục trước khi chọn ngưỡng. Accuracy có thể ghi phụ cho E1; không dùng accuracy đơn lẻ để kết luận E2 vì tỷ lệ Bad rất thấp.

Nếu so các model, chỉ so cùng tập test, nhãn dương, cách tiền xử lý và đơn vị account. Seed khác nhau là lần chạy khác nhau. Đối với E2, cân bằng lớp hoặc lọc hoạt động phải được mô tả riêng cho train; validation/test dùng cùng quy tắc đánh giá đã chốt, không lấy mẫu giảm để làm đẹp điểm. Nếu có lọc validation/test theo hoạt động như proposal, phải báo rõ số tài khoản trước/sau lọc và dùng đúng cùng tập lọc cho mọi model.

## Chỉ số graph E3

Người làm E3 cần bàn giao: danh sách đỉnh kể cả đỉnh cô lập, edge list có trọng số, quy tắc tạo cạnh, thời gian quan sát, thuật toán/phân giải/seed tìm cộng đồng, community assignment và metrics CSV. Báo số nút/cạnh, số thành phần, LCC, density, tỷ lệ đỉnh cô lập, số/kích thước cộng đồng, modularity nếu thuật toán cung cấp. Chỉ so modularity, ARI hoặc Bad enrichment khi các graph thật sự dùng cùng tập đỉnh và cách tìm cộng đồng. Bad ratio/coverage/enrichment chỉ có ý nghĩa phân tích gián tiếp, không phải độ chính xác campaign.

## Schema `baseline_results.csv`

Mỗi dòng là **một chỉ số đã tính**: `experiment,run_id,model,split,seed,positive_label,metric,value,n_accounts,input,feature_set,notes`. `input` là file prediction hoặc graph/metrics gốc đã sinh số; `notes` dùng cho ngưỡng và giới hạn. Chỉ ghi số thực, không dùng `0` hoặc `NA` để thay kết quả chưa chạy. Nếu chưa nhận thí nghiệm, giữ bảng tổng hợp ở trạng thái “chưa nhận”.
