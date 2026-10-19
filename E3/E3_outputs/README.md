# E3 — Coordination Graph Baseline
## Chạy lại
1. Mở E3_Coordination_Graph.ipynb trên Colab, dùng CPU runtime.
2. Chạy cell 1 và hoàn tất Google Drive mount bằng tài khoản có quyền đọc MXH.
3. DATA ở cell 2 trỏ đến MXH/processed qua shortcut trong MXH_4_output.
   Có thể sửa DATA nếu shortcut nằm ở vị trí khác. OUT hiện tại: /content/drive/MyDrive/Colab Notebooks/MXH_4_output/E3_outputs
4. Chạy lần lượt cell 2 → 7.
5. Kiểm tra bảng metrics, community summary và e3_manual_review.csv.
   Điền review_status/review_note sau khi thực sự đọc evidence.
6. Lưu notebook bằng Ctrl+S. Outputs riêng tư của Colab có thể không nằm trong ipynb;
   các CSV/JSON/PNG được lưu riêng ở OUT.

## Input
Hai Honduras normalized JSONL, honduras_account_manifest.csv,
honduras_account_splits.csv, quality_audit/group_b_metadata_conflicts.csv,
quality_audit/group_c_text_conflicts.csv. Không cần Fox8 hay feature table E2.
Schema: account_id, tweet_id, label, text, created_at, hashtags.
Label 0=Bad, 1=Good; nhãn chỉ dùng mô tả hậu nghiệm, không dùng tạo graph.
File gốc chỉ đọc. Conflict IDs B/C bị loại toàn bộ; ID còn lại giữ occurrence đầu.

## Định nghĩa graph
Node: mọi account trong manifest, giữ cả isolate.
Event: một hashtag NFKC/casefold, bỏ # đầu, trong giờ UTC cố định 3600 giây.
Mỗi account chỉ tính một lần/event. Event có 2–100 account mới được dùng.
Edge vô hướng: hai account cùng event; weight là số event khác nhau dùng chung.
Giữ weight >= 2, không self-loop. Không giới hạn số tweet theo label.
Louvain: NetworkX, weighted, resolution=1, seed=42, chạy trên active nodes.
Community -1 trong node table là isolate, không phải một community thật.

## Output
- e3_graph_edges.csv: source,target,weight
- e3_graph_nodes.csv: account_id,label,split,degree,strength,community
- e3_graph_metrics.csv: node/edge/isolate/component/community statistics, modularity
- e3_communities.csv, e3_components.csv: bảng thống kê
- e3_data_quality.csv, e3_event_audit.csv: số record bị loại và event bị bỏ
- e3_manual_review.csv: evidence của cạnh mạnh nhất trong tối đa 3 community lớn
- e3_community_preview.png: hình một phần community lớn nhất nếu graph có cạnh
- e3_run_config.json: cấu hình, SHA256 input và phiên bản runtime
- requirements_e3.txt: phiên bản package của lần chạy

## Giới hạn và cách diễn giải
Đây là baseline shared-hashtag coordination/community discovery, chưa xác nhận campaign.
Không có ground-truth campaign labels nên không báo Precision/Recall/F1 campaign.
Modularity và bad_ratio chỉ là mô tả, không chứng minh phối hợp độc hại.
Hashtag phổ biến, tin nóng, retweet cùng nội dung và chênh lệch mức hoạt động có thể tạo cạnh tự nhiên.
Giờ cố định có boundary effect; cùng hashtag ở hai giờ liền kề không nối nhau.
Ngưỡng event tối đa bỏ cả event phổ biến, làm mất một phần tín hiệu; audit ghi rõ số bị bỏ.
Threshold được khai báo trước, không chọn theo label/test. Chưa thực hiện sensitivity analysis.
Graph dùng toàn corpus (transductive/mô tả), giữ split làm metadata, không thay split E1/E2.
Không dùng graph này cho đánh giá inductive E2 nếu chưa thiết kế lại quy trình tránh leakage.
Nếu graph rỗng: báo đúng 0 cạnh; không âm thầm giảm ngưỡng để tạo kết quả đẹp.
Nếu vượt pair budget: chương trình dừng, không xuất graph cắt cụt; sửa cấu hình và chạy lại.

## Reproducibility checklist
[x] Kiểm tra số account và label, manifest/split không giao nhau theo account.
[x] Không dùng label hoặc split để chọn cạnh.
[x] Duplicate policy và input hashes được ghi lại.
[x] Seed và thứ tự node/edge được cố định.
[x] Reload output kiểm tra ID, degree, split, label.
[ ] Người thực hiện xác nhận review thủ công trong CSV.
[ ] Thành viên khác chạy toàn bộ trên runtime sạch và đối chiếu output.

## Assisted evidence review
Xem E3_RESULTS.md và e3_manual_review.csv. Ba cặp/18 dòng đã được đọc; reviewed_evidence_only không phải nhãn malicious. Human validation vẫn cần nếu đưa ra kết luận campaign.
