# E3 — Quyết định λ, top-k và cosine threshold

Quyết định: λ=0.5, top-k=5, cosine threshold=0.9, dùng chung cho Scratch và Transfer.

## 1. Mục tiêu và dữ liệu

Chọn graph thưa, giữ độ phủ và khả năng liên thông; không tối ưu độ chính xác campaign vì không có campaign ground truth. Dùng 22.468 account Validation cố định, trong đó 22.465 có tín hiệu TF-IDF. Ba account không có tín hiệu vẫn giữ ở graph chính, không tạo similarity từ bias-only embedding. Không đọc nhãn Bad/Good hoặc dùng kết quả Test để xếp hạng cấu hình.

Lượt pilot thực hiện sau lượt thăm dò full corpus với 0.5/10/0.8. Quy tắc lựa chọn được ghi vào selection_rule.json trước khi chạy bảng pilot; đây không phải preregistration trước toàn bộ nghiên cứu.

## 2. Lưới và quy tắc đã khai báo trước pilot

Thử threshold={0.7,0.8,0.9}, k={5,10,20}: 9 cấu hình/encoder, 18 dòng kết quả. L2-normalize embedding; dùng exact cosine search trên Validation, bỏ self-loop, đối xứng union/max. Với nhiều neighbor đồng điểm, exact search có thể trả một tập tied candidates; không coi ID của chúng là duy nhất.

Chọn threshold lớn nhất có ít nhất 90% valid nodes có cạnh ở cả hai encoder. Sau đó chọn k nhỏ nhất vẫn phủ ít nhất 90% valid nodes và giữ ít nhất 95% LCC coverage so với k=20, ở cả hai encoder. Các mức 90%/95% là ràng buộc thiết kế được công bố trước lượt pilot, không phải ngưỡng phổ quát của coordination detection.

## 3. Lý do threshold=0.9

Cả ba threshold đều đạt yêu cầu. Chọn 0.9 vì đây là mức nghiêm ngặt nhất trong lưới vẫn đạt độ phủ. Tuy nhiên, các top-neighbor cosine rất cao nên threshold 0.7–0.9 hầu như không thay đổi graph. Không kết luận 0.9 là ngưỡng ngữ nghĩa tối ưu hoặc bằng chứng phối hợp.

## 4. Lý do top-k=5

k=5 đã đạt ràng buộc LCC retention ở cả hai encoder và vẫn giữ 100% valid-node coverage. Chọn mức nhỏ nhất đạt yêu cầu để giảm số cạnh và chi phí so với k=10/20. Sau đối xứng hóa, degree có thể lớn hơn 5; k chỉ giới hạn lựa chọn láng giềng có hướng trước union.

| method   |   threshold |   top_k |   n_nodes |   n_edges |   active_coverage |   lcc_fraction |   median_degree |   max_degree |   weight_sum |   lcc_retention_vs_k20 |   edge_reduction_vs_k20 |
|:---------|------------:|--------:|----------:|----------:|------------------:|---------------:|----------------:|-------------:|-------------:|-----------------------:|------------------------:|
| scratch  |         0.9 |       5 |     22465 |     87238 |                 1 |       0.945426 |               6 |          136 |      87206.5 |               0.954562 |                0.736605 |
| transfer |         0.9 |       5 |     22465 |     84062 |                 1 |       0.967371 |               6 |          236 |      84025   |               0.967371 |                0.733411 |

## 5. Lý do lambda=0.5 và thay đổi quy ước scale

Không có cơ sở độc lập để ưu tiên baseline hay similarity theo chất lượng campaign. Chọn lambda=0.5 là thiết kế trung lập về tổng trọng số, không phải giá trị được chứng minh tối ưu. Không dùng modularity hoặc Bad Ratio trên full corpus để chọn lambda.

Trước khi trộn, chuẩn hóa riêng từng graph bằng tổng trọng số cạnh vô hướng dương:

W_base_hat = W_base / sum_{u<v} W_base[u,v]

W_sim_hat = W_sim / sum_{u<v} W_sim[u,v]

W_enhanced = (1-lambda)*W_base_hat + lambda*W_sim_hat.

Chỉ scale trọng số, không thay topology baseline. Nếu một nguồn có tổng trọng số bằng 0, dừng và báo lỗi thay vì chia 0 hoặc âm thầm đổi công thức. Với lambda=0.5, mỗi nguồn đóng góp một nửa tổng trọng số trước khi cộng cạnh trùng. Không có nghĩa là mỗi nguồn đóng góp một nửa chất lượng phát hiện.

Giữ trọng số baseline chưa hiệu chuẩn rồi trộn với cosine gần 1 sẽ khiến similarity chi phối, dù cả hai đều nằm trong [0,1]. Audit trên Validation:

| method   |   validation_baseline_weight_sum |   validation_similarity_weight_sum |   similarity_mass_fraction_without_calibration_at_lambda_05 |   similarity_mass_fraction_after_sum_normalization_at_lambda_05 |
|:---------|---------------------------------:|-----------------------------------:|------------------------------------------------------------:|----------------------------------------------------------------:|
| scratch  |                          12.8867 |                            87206.5 |                                                    0.999852 |                                                             0.5 |
| transfer |                          12.8867 |                            84025   |                                                    0.999847 |                                                             0.5 |

## 6. Áp dụng và trạng thái

Dùng cùng λ/k/threshold, chuẩn hóa tổng trọng số và Leiden seed=42, resolution=1, RBConfigurationVertexPartition, iterations=-1 cho cả hai graph tăng cường. Scope chính giữ full corpus, mô tả/transductive, cùng node set với người 4. Full-corpus ANN cần được audit riêng; pilot exact search không chứng minh ANN full-corpus là chính xác.

Thay đổi cả k/threshold và weight normalization so với lượt thăm dò cũ. Không đổi tên output cũ thành kết quả cấu hình mới. Cần chạy lại hai graph và Leiden với cấu hình đã chọn; lưu tại selected_run/. Lượt thăm dò chưa hoàn tất đã được ngắt, embedding và baseline kiểm tra vẫn được giữ.

## 7. Giới hạn và sensitivity

Tiêu chí trên chọn sparsity/coverage, không xác nhận semantic coordination, bot hay malicious campaign. Cosine gần 1 và nhiều tied neighbors cần được nêu rõ. Chưa chạy sensitivity của lambda=0.25/0.75, edge-count control hay kiểm định community stability; không ghi các phân tích này là đã hoàn tất.

Vấn đề pos_weight và validation checkpoint của E2 vẫn cần người 3 xác minh. Tham số graph này không sửa được hạn chế của checkpoint đầu vào.

## Tái chạy

Chạy cell pilot rồi cell quyết định này trước cell tạo graph chính. Có thể đọc toàn bộ 18 cấu hình từ validation_grid.csv; config chính nằm ở chosen_config.json.