# Thiết lập thí nghiệm

E1 chọn checkpoint bằng Macro-F1 trên Dev, bộ chuẩn hóa đặc trưng số được fit trên Train. Checkpoint đã nạp strict thành công và nhánh tweet trả về 128 chiều.

E2 cố định split gốc và lọc selected_tweet_count > 0. Train gồm 1.209 Bad và 34.246 Good, Validation gồm 172 Bad và 3.469 Good, Test gồm 345 Bad và 7.383 Good. Class weights tính từ Train. Ba seed là 42, 52, 62. Chọn checkpoint theo Validation Macro-F1, dùng Validation Loss để phân xử khi bằng điểm.

E3 giữ 224.685 node. Cấu hình bàn giao dùng cosine ≥ 0,90, top-k 5 và Leiden resolution 1, seed 42. Graph A không có cạnh đạt yêu cầu xác thực. Graph B bàn giao dùng similarity.

SHA-256 của checkpoint E1 trên đĩa khớp cấu hình chuyển giao E2. Phép kiểm E3 dùng tìm cosine chính xác theo khối, loại self-loop theo ID và chạy Leiden seed 42. Khi có điểm bằng nhau, thứ tự top-k có thể khác lượt FAISS công bố.
