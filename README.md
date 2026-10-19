# SNA-GROUP3
bot detection on Fox8-23, Bad/Good classification and coordination graph analysis on Honduras.

## Cấu trúc

| Thư mục | Nội dung |
| --- | --- |
| `data/preprocessing/` | Khảo sát, chuẩn hóa, manifest, split và audit duplicate |
| `data/validation/` | Xác minh nhãn Honduras và tạo feature cấp tài khoản |
| `E1/` | Bot/Human baseline trên Fox8-23 |
| `E2/` | Bad/Good baseline trên Honduras |
| `E3/` | Graph hành vi và phân tích cộng đồng |
| [evaluation](evaluation/README.md) | Đánh giá, tổng hợp và kiểm tra tái lập |

Các thư mục data và E1/E2/E3 hiện dùng `.gitkeep` để giữ cấu trúc; code chưa được tích hợp vào các thư mục này. Khi tích hợp từng phần, cập nhật tại README này tên notebook/script, thứ tự chạy, phiên bản thư viện và vị trí input/output. Nếu phần nào cần hướng dẫn dài, có thể bổ sung README riêng khi cần.

Dữ liệu gốc và model lớn lưu ngoài repo; ghi nguồn tải và đường dẫn đặt file trong hướng dẫn chạy. Giữ split tài khoản đã thống nhất, không tự chia lại trong từng experiment.

Kết quả hiện có là baseline, chưa chứng minh hiệu quả transfer learning. Hướng dẫn kiểm tra hiện tại và giới hạn nằm trong [evaluation/README.md](evaluation/README.md); các đường dẫn local trong script còn cần cập nhật khi tích hợp.

Quy ước nhánh, commit và Pull Request: [CONTRIBUTING.md](CONTRIBUTING.md). Mỗi phần làm trên nhánh riêng rồi tạo PR vào `main`.
