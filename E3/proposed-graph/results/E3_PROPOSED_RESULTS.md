## Backend CPU

Dùng igraph + leidenalg, RBConfigurationVertexPartition, seed=42, resolution=1, n_iterations=-1 theo người 4. Không dùng cuGraph. Output cấu hình đã chọn lưu ở selected_run/.

# E3_PROPOSED_RESULTS — Người 5

## Cấu hình đã chọn và lý do

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

## Kết quả graph chính

| method                 |   n_nodes |   n_active_nodes |   n_edges |   isolated_node_rate |   largest_component_nodes |   n_communities |   modularity |
|:-----------------------|----------:|-----------------:|----------:|---------------------:|--------------------------:|----------------:|-------------:|
| baseline_delivered     |    224685 |            86488 |   1922134 |          0.61507     |                     81055 |            2127 |     0.67903  |
| baseline_plus_scratch  |    224685 |           224640 |   2853967 |          0.00020028  |                    220826 |             742 |     0.736721 |
| baseline_plus_transfer |    224685 |           224639 |   2834812 |          0.000204731 |                    220417 |             698 |     0.739346 |

## Thay đổi cấu trúc

| method                 |   n_edges_added |   n_edges_removed |   similarity_baseline_overlap |   active_partition_ari_vs_baseline |
|:-----------------------|----------------:|------------------:|------------------------------:|-----------------------------------:|
| baseline_delivered     |             nan |               nan |                           nan |                         nan        |
| baseline_plus_scratch  |          931833 |                 0 |                         11319 |                           0.373404 |
| baseline_plus_transfer |          912678 |                 0 |                         10597 |                           0.357068 |

## Audit similarity/ANN full corpus

| method   |   n_accounts |   n_valid_embedding_accounts |   embedding_coverage |   n_similarity_edges |   ann_sample_size |   ann_id_recall_at_k |   ann_kth_score_success_rate | ann_note                                                                                  |
|:---------|-------------:|-----------------------------:|---------------------:|---------------------:|------------------:|---------------------:|-----------------------------:|:------------------------------------------------------------------------------------------|
| scratch  |       224685 |                       224635 |             0.999777 |               943152 |               128 |             0.742188 |                     0.945312 | ID recall can be low with tied/identical vectors; score agreement is reported separately. |
| transfer |       224685 |                       224635 |             0.999777 |               923275 |               128 |             0.70625  |                     0.914062 | ID recall can be low with tied/identical vectors; score agreement is reported separately. |

## Provenance và giới hạn

Embedding tái tạo bằng checkpoint E2 và TF-IDF E1, đối chiếu khớp toàn bộ 44.938 embedding Test của người 3. Baseline CPU đã được tái tạo khớp bản bàn giao: ARI=1 và modularity khác khoảng 1e-11. Giữ bản bàn giao làm đối chứng CPU, không chạy lại phép kiểm tra dài này.

Account representation học từ nội dung TF-IDF qua MLP, chưa gồm toàn bộ hành vi thời gian/hành động. Giữ mọi node; account TF-IDF rỗng không tạo similarity từ bias-only embedding. Full corpus gồm Train accounts của E2 nên đây là mô tả/transductive, không phải held-out campaign detection.

Không dùng Bad/Good để tạo cạnh/chọn tham số. Modularity hay nhiều cạnh hơn không chứng minh phát hiện campaign tốt hơn. Chưa có edge-count control hoặc lambda sensitivity; không kết luận Transfer vượt Scratch chỉ từ graph metrics.

E2: code Good=1 nhưng pos_weight=n_Good/n_Bad đang tăng trọng số Good; người 3 cần xác minh. Code bàn giao train 10 epoch, chưa thể hiện validation checkpoint selection. Giữ nguyên checkpoint, không âm thầm sửa.

## Output

Embeddings và account IDs ở thư mục cha. parameter_selection/ lưu toàn bộ bước chọn tham số. selected_run/ lưu similarity graphs, enhanced graphs, node/community assignments, metrics, bảng/hình so sánh và run_config.json.

## Activity-level analysis (hậu nghiệm)
| method                 | activity_source                                           |   spearman_tweet_count_degree |   spearman_tweet_count_strength |   high_activity_cutoff |   high_activity_account_fraction |   high_activity_degree_share |   high_activity_strength_share |   degree_top1pct_share |
|:-----------------------|:----------------------------------------------------------|------------------------------:|--------------------------------:|-----------------------:|---------------------------------:|-----------------------------:|-------------------------------:|-----------------------:|
| baseline_delivered     | all normalized records incl. duplicates, descriptive only |                      0.529802 |                        0.532935 |                      9 |                         0.103977 |                     0.637197 |                       0.692162 |               0.29636  |
| baseline_plus_scratch  | all normalized records incl. duplicates, descriptive only |                      0.553621 |                        0.508507 |                      9 |                         0.103977 |                     0.462845 |                       0.398396 |               0.22275  |
| baseline_plus_transfer | all normalized records incl. duplicates, descriptive only |                      0.547711 |                        0.498557 |                      9 |                         0.103977 |                     0.463838 |                       0.395966 |               0.224773 |
Activity = số record trong normalized corpus trước conflict/dedup filtering, không phải số event độc lập. Nhóm >= phân vị 90 có thể chiếm hơn 10% node do ties. Correlation/degree concentration không chứng minh nguyên nhân.
