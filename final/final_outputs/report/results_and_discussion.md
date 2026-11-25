# Kết quả và thảo luận

## E1

Tính lại từ 1.183 dự đoán Test cho Accuracy 0,841927 và Macro-F1 0,840446. Kết quả khớp artifact. Checkpoint TwiBot-20 mới và thí nghiệm Fox8 cũ có dữ liệu và thiết kế khác nhau.

## E2

Accuracy và Bad F1 tính lại từ cả chín ma trận nhầm lẫn đều khớp. Trên 7.728 tài khoản Test có tweet, Scratch đạt Macro-F1 0,9568 và Bad F1 0,9174. Transfer đạt 0,9546 và 0,9132, với Bad AP 0,9599 so với 0,9585 của Scratch. Frozen Transfer đạt Macro-F1 0,8606 và Bad F1 0,7345. AP trong bảng bàn giao là trung bình theo seed. E2 chưa cho thấy cải thiện rõ về F1 từ chuyển giao trong thiết lập này.

## E3

Graph A có 0 cạnh được xác thực. Graph B bàn giao có 995.347 cạnh, 2.084 community và weighted modularity 0,1976. Tất cả 177.861 tài khoản thiếu tweet được gán community. Trong đó 177.522 ở community 0 và 339 ở community 4. Đầu vào toàn zero cho vector sau tweet_branch có norm 8,47 khi chạy checkpoint. Điều kiện lọc norm sau phép chiếu vì vậy giữ lại đầu vào thiếu tweet. Tỷ lệ isolate 5,76% và độ phủ Bad 80,49% của bản bàn giao bao gồm hiệu ứng này.

## Kiểm định lại bằng checkpoint

E2 chạy inference CPU từ cả chín checkpoint trên 7.728 tài khoản Test. Accuracy, Macro-F1 và Bad F1 khớp tuyệt đối ở mọi seed. Sai khác Bad AP tối đa 1,94 × 10⁻⁷. Dự đoán từng tài khoản đã lưu trong audit.

## E3 sau khi lọc tài khoản thiếu tweet

Graph chỉ dùng similarity giữ 224.685 node, có 106.061 cạnh, 33.874 node có cạnh và 2.137 community. Modularity là 0,8191, thành phần liên thông lớn nhất có 28.164 node và community lớn nhất có 3.577 node. Có 190.811 node cô lập, tương ứng 84,92% toàn bộ node. 1.362/1.866 Bad toàn corpus có cạnh (72,99%), hay 1.362/1.726 Bad có tweet (78,91%). Cả 177.861 tài khoản thiếu tweet đều cô lập theo thiết kế. Q thay đổi khi tập cạnh và tập node hoạt động thay đổi. Kết quả dùng để kiểm tra độ nhạy của cách dựng graph.

## Kiểm tra độ nhạy trên toàn bộ Test E2

Chạy lại chín checkpoint trên cả 44.938 tài khoản Test cho Bad F1 trung bình 0,88118 với Scratch, 0,87699 với Transfer và 0,01575 với Frozen Transfer. Trên 37.210 tài khoản thiếu tweet, Scratch và Transfer dự đoán toàn bộ là Good, còn Frozen Transfer dự đoán toàn bộ là Bad ở cả ba seed. Đây là đánh giá bổ sung trên đầu vào zero ngoài phạm vi huấn luyện chính. Bảng kết quả chính tiếp tục dùng 7.728 tài khoản có tweet. Bằng chứng nằm trong audit/e2_cohort_sensitivity.csv.

## Phân bố Bad sau khi lọc graph E3

Trong 2.137 community của đồ thị đã lọc, 24 community chứa ít nhất một tài khoản Bad. Có 504 tài khoản Bad cô lập. Bảng tables/e3_masked_community_profiles.csv liệt kê số Bad, Good và Bad Ratio theo community. Các nhãn được ghép sau khi phân cụm.

## Kiểm tra phiên bản và hiện vật

Clone mới của nhánh final khớp commit ccc53ac và cả ba cây Git E1, E2, E3 với checkout kiểm định. SHA-256 checkpoint E1 là 219c21ca894c6ea4bcd8a4ac009f0383880892aa0df665795e0ef8f880ce7394, khớp cấu hình nguồn chuyển giao E2. Chín epoch tốt nhất khớp lịch sử validation. Đồ thị có lọc không có cạnh trùng, self-loop hoặc đầu mút thiếu tweet. Chi tiết trong audit/handoff_checks.json.

## Huấn luyện lại E2 trên CPU

Đã huấn luyện lại ba cấu hình qua ba seed theo notebook, với đầu vào Honduras và checkpoint E1 bàn giao. Bad F1 trung bình Test lần lượt là 0,91953 (Scratch), 0,90979 (Transfer) và 0,73453 (Frozen Transfer). Macro-F1 lần lượt là 0,95789, 0,95282 và 0,86062. Đây là kết quả huấn luyện lại riêng, bên cạnh bảng kiểm tra checkpoint bàn giao. Các lịch sử, checkpoint và bảng từng seed nằm trong audit/e2_retraining.

## E1 và E3 sau khi huấn luyện lại E1

E1 đã tái tạo cả bốn nhóm đặc trưng và huấn luyện lại từ dữ liệu TwiBot-20. Checkpoint tốt nhất ở epoch 5, đạt Dev Macro-F1 0,8445. Trên 1.183 tài khoản Test, Accuracy là 0,841927 và Macro-F1 là 0,838830. Checkpoint bàn giao đạt cùng Accuracy và Macro-F1 0,840446. Hai lượt có ma trận nhầm lẫn khác nhau.

Dùng checkpoint E1 vừa huấn luyện để dựng lại Graph B với cùng điều kiện lọc tweet, cosine và Leiden. Số cạnh giảm từ 106.061 xuống 69.877, số community tăng từ 2.137 lên 2.798. Có 27.690 node có cạnh, gồm 1.258 Bad. Tỷ lệ cô lập là 87,68%, weighted modularity 0,8758. Chênh lệch cho thấy cấu trúc Graph B nhạy với checkpoint E1. Audit E2 huấn luyện lại sử dụng checkpoint E1 bàn giao.

## Huấn luyện lại E2 với checkpoint E1 mới

Sau khi huấn luyện lại E1, nhóm kiểm định chạy lại toàn bộ E2 với ba cấu hình và ba seed 42, 52, 62. Cùng tập Test 7.728 tài khoản, Bad F1 trung bình lần lượt là 0,91953 cho Scratch, 0,91645 cho Transfer và 0,77304 cho Frozen Transfer. Khi dùng checkpoint E1 bàn giao, các số tương ứng là 0,91953, 0,90979 và 0,73453. Scratch không nạp trọng số E1 và giữ nguyên kết quả. Hai cấu hình chuyển giao có Bad F1 cao hơn ở lượt dùng checkpoint mới. Scratch vẫn có Bad F1 trung bình cao nhất trong lượt này. Kết quả theo từng seed và chênh lệch nằm trong audit/e2_from_retrained_e1 và audit/e2_e1_checkpoint_comparison.csv.
