# Phương pháp kiểm định

Bản kiểm định dùng nhánh final tại ccc53ac. E1 trên TwiBot-20 tạo bộ mã hóa đa nhánh từ bài đăng, hồ sơ và hàng xóm. E2 dùng MiniLM 384 chiều từ Honduras rồi so Scratch, Transfer và Frozen Transfer. E3 đưa cùng embedding qua tweet_branch của E1 để tạo vector 128 chiều. Nhãn Bad/Good chỉ dùng để mô tả phân bố sau phân cụm.

Đối chiếu kiểm định theo ba mức: đọc code, tính lại từ artifact, và chạy lại checkpoint. Với E3, phép phân tích độ nhạy giữ toàn bộ node và chỉ xét tài khoản có tweet hợp lệ ở cả hai đầu mỗi cạnh similarity.

## Khảo sát dữ liệu và kiểm tra đặc trưng

E1 có 11.826 tài khoản trên ba split, gồm Train 3.632 Human và 4.646 Bot, Dev 1.062 Human và 1.303 Bot, Test 543 Human và 640 Bot. Bốn mảng được tái tạo có số chiều theo thứ tự tweet, hồ sơ văn bản, hồ sơ số và Boolean, hàng xóm là 384, 384, 15 và 768. Toàn bộ giá trị hữu hạn. Có 80 dòng tweet toàn zero và 1.273 dòng hàng xóm toàn zero.

Honduras có 1.866 Bad trên 224.685 tài khoản (0,83%). Ma trận MiniLM 224.685 × 384 không chứa NaN hoặc Inf. Có 177.861 dòng toàn zero, gồm 140 Bad và 177.721 Good theo báo cáo validation. Tỷ lệ thiếu tweet chênh lệch mạnh theo nhãn. Nhóm giữ đủ tài khoản trong metadata. E2 đánh giá chính trên tập con có tweet, còn phép kiểm bổ sung chấm toàn bộ Test. E3 giữ đủ node và dùng bộ lọc tweet để tránh tạo cạnh từ đầu vào zero. Bằng chứng kiểm tra trực tiếp các mảng nằm trong audit/feature_eda.json.
