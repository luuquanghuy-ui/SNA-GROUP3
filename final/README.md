This dataset, named Twibot-20, is a comprehensive sample of the Twittersphere.

This dataset focuses on bot detection on Twitter.

The datasets are released under MIT(https://github.com/GabrielHam/TwiBot-20/blob/main/LICENSE)

The train, validation, test and support set are provided in .json format.

The user in support set provides the neighborhood information.

Each user sample contains:
- 'ID': the ID from Twitter identifying the user.
- 'profile': the profile information obtained from Twitter API.
- 'tweet': the recent 200 tweets of this user.
- 'neighbor': the random 20 followers and followings of this user.
- 'domain': the domain of this user and the domains include politics, business, entertainment and sports.
- 'label': the label of this user and '1' means it is a bot while '0' means it is a human.
## E1 – Tiền xử lý Support và xây dựng đặc trưng Neighbor

### 1. Mục đích sử dụng Support

Trong E1, nhóm sử dụng `support.json` làm nguồn dữ liệu bổ trợ để khai thác thông tin mạng lưới của các tài khoản thuộc tập train, validation và test.

Support set cung cấp thông tin hồ sơ của những tài khoản được tham chiếu trong danh sách follower và following của tài khoản mục tiêu.

### 2. Quy trình xử lý Support

Do `support.json` có dung lượng lớn (khoảng 4.9 GB), nhóm thực hiện trích xuất có chọn lọc thay vì nạp toàn bộ dữ liệu vào bộ nhớ:

- Đọc ID của các tài khoản hàng xóm được tham chiếu trong train, validation và test.
- Đối chiếu các ID này với dữ liệu trong `support.json` để tìm hồ sơ tương ứng.
- Trích xuất các thuộc tính hồ sơ gồm `name`, `screen_name`, `description` và `location`.
- Lưu hồ sơ hàng xóm đã trích xuất vào `support_neighbors.jsonl`.
- Lưu quan hệ giữa tài khoản mục tiêu và các tài khoản hàng xóm vào `neighbor_edges.jsonl`.

Cách xử lý này giúp giảm dung lượng dữ liệu cần sử dụng, hạn chế tiêu thụ bộ nhớ và tránh phải đọc lại toàn bộ support set trong các bước tiếp theo.

### 3. Xây dựng đặc trưng Neighbor

Thông tin hồ sơ của các tài khoản hàng xóm được mã hóa bằng mô hình `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

Các vector biểu diễn thu được được sử dụng để xây dựng đặc trưng Neighbor, bổ sung thông tin mạng lưới xung quanh tài khoản mục tiêu.

Trong kiến trúc E1, đặc trưng Neighbor là một nhánh đầu vào riêng, kết hợp với hai nhánh biểu diễn thông tin hồ sơ (Profile) và nội dung bài đăng (Tweet).

Nhờ đó, mô hình có thể xem xét cả đặc điểm cá nhân của tài khoản lẫn thông tin từ các tài khoản có liên kết với nó.

### 4. Lưu ý

- `support.json` là dữ liệu nguồn ban đầu, không được sử dụng như một tập dữ liệu gán nhãn bổ sung.
- `support_neighbors.jsonl` chỉ chứa các hồ sơ hàng xóm được trích xuất có chọn lọc.
- `neighbor_edges.jsonl` lưu thông tin liên kết giữa tài khoản mục tiêu và các tài khoản hàng xóm.
- Những tài khoản hàng xóm không tìm thấy hồ sơ tương ứng trong support set được xử lý như trường hợp thiếu dữ liệu trong quá trình chuẩn bị đặc trưng.
