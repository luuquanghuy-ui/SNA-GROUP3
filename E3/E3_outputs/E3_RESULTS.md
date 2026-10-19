# E3 — Kết quả baseline
Dữ liệu: 1,262,830 records ban đầu; giữ 1,254,185 tweet hợp lệ sau lọc.
Loại 1,362 rows của 389 conflict IDs và 7,283 rows trùng còn lại.
Node: 224,685; active: 84,819; isolate: 139,866.
Cạnh: 1,905,386; components: 141,752; cộng đồng active: 2,101.
Modularity weighted: 0.6540762661863168; community lớn nhất: 17,482 node.
Bỏ 3,536 event có hơn 100 account.
Graph dùng hashtag–giờ UTC, minimum weight 2; Louvain seed 42.

## Review mẫu đã đọc

('1071499269880901639', '955569674737127425'): Đã đọc 3 event mẫu: hai account chia sẻ nội dung của ChalecosAmarill; một event cùng timestamp ghi nhận, một lệch 60 giây, một lệch 40 phút. Có shared amplification nhưng chưa chứng minh campaign độc hại.

('141160059', '145459615'): Đã đọc 3 event mẫu: hai event có nội dung DolarToday giống/gần giống dạng original–retweet; event #10nov ở bucket 437058 có text khác chủ đề. Hashtag ngày tháng có thể tạo cạnh yếu về ngữ nghĩa.

('149310384', '311680564'): Đã đọc 3 event #aliado: nội dung tin dầu mỏ gần giống nhau, cùng timestamp ghi nhận ở mỗi event, khác URL rút gọn. Phù hợp giả thuyết đăng lại/syndication; không đủ kết luận malicious coordination.

## Diễn giải và giới hạn
Một tweet có nhiều hashtag có thể đóng góp nhiều event cho cùng cặp; weight KHÔNG phải số lần phối hợp độc lập.
Hashtag ngày tháng/tin nóng và nguồn tin dùng chung có thể tạo nhiều cạnh tự nhiên.
Timestamp chỉ chính xác ở mức dữ liệu ghi nhận; cùng timestamp không chứng minh cùng giây đăng thực tế.
Ba cặp được chọn từ ba community lớn nhất, cạnh mạnh nhất, không đại diện ngẫu nhiên cho toàn graph.
Đã kiểm tra evidence, chưa có xác minh người dùng thật hay ground-truth campaign.
Bad/Good là nhãn account, không phải nhãn campaign. Không diễn giải modularity thành độ chính xác.
Chưa thực hiện transfer-learning graph, ablation, sensitivity analysis hoặc đánh giá classifier.
