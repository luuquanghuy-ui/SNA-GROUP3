# Quy ước làm việc trong repository

## Thư mục phụ trách

| Thư mục | Nội dung |
| --- | --- |
| `data/preprocessing/` | Khảo sát, chuẩn hóa, manifest, split và audit duplicate |
| `data/validation/` | Xác minh nhãn Honduras và tạo feature cấp tài khoản |
| `E1/` | Fox8-23: notebook, code và hướng dẫn E1 |
| `E2/` | Honduras Bad/Good: notebook, code và hướng dẫn E2 |
| `E3/` | Coordination graph: notebook, code và hướng dẫn E3 |
| `evaluation/` | Tổng hợp metric, kiểm tra tái lập và báo cáo đối chiếu |

Mỗi phần cập nhật hướng dẫn chạy trong README gốc, ghi input/output và thứ tự chạy. Chỉ thêm README riêng nếu phần đó cần hướng dẫn chi tiết. File dùng chung hoặc thay đổi giao diện đầu vào phải được mô tả trong PR. Không đưa dữ liệu gốc, checkpoint lớn, môi trường ảo, cache hoặc thông tin đăng nhập vào commit.

## Branch

Mỗi công việc tạo một nhánh từ `main` mới nhất. Đặt tên theo dạng `data/preprocessing-<mo-ta>`, `data/validation-<mo-ta>`, `e1/<mo-ta>`, `e2/<mo-ta>`, `e3/<mo-ta>`, `evaluation/<mo-ta>` hoặc `docs/<mo-ta>`. Nhánh `evaluation-huy` đã dùng là hợp lệ; không cần đổi tên lịch sử.

Ví dụ sau khi clone repo và khi không có thay đổi chưa lưu:

```powershell
git switch main
git pull --ff-only origin main
git switch -c e1/add-baseline
```

Làm việc trên nhánh riêng và chỉ stage file thuộc thay đổi đang làm. Mỗi người cấu hình tên/email Git của mình; email cần gắn với tài khoản GitHub để hỗ trợ ghi nhận đúng tác giả commit.

## Commit và push

Commit ngắn gọn, ghi rõ nội dung; một commit nên phục vụ một thay đổi có thể hiểu được. Ví dụ:

```powershell
git status --short
git add E1
git commit -m "Add E1 baseline notebook and run instructions"
git push -u origin e1/add-baseline
```

Kiểm tra danh sách file trước khi commit. Không dùng commit để ghi kết quả chưa chạy hoặc đổi số liệu cho khớp kỳ vọng.

## Pull Request

Tạo PR từ nhánh làm việc vào `main`; tiêu đề mô tả thay đổi. Nội dung ghi các file/phần thay đổi, input/output, kiểm tra đã chạy và giới hạn còn lại. Khuyến khích một thành viên khác xem trước khi merge.

Trước khi merge, kiểm tra tab Files changed, kết quả kiểm tra đã ghi và tình trạng xung đột. Không ghi đè phần của thành viên khác để giải quyết conflict. Nếu cần cập nhật `main` vào nhánh đang làm, commit công việc hiện tại trước rồi chạy:

```powershell
git fetch origin
git merge origin/main
```

Nếu có conflict, mở từng file và giữ nội dung phù hợp sau khi đối chiếu. Sau khi giải quyết và kiểm tra lại, `git add` các file đã sửa, `git commit`, rồi `git push`. Nếu merge thành công ngay, chỉ cần push. Khi chưa chắc cách giải quyết, có thể `git merge --abort` để hủy lần merge đang xung đột và trao đổi với người phụ trách.

Sau khi PR được merge, lấy bản mới trên `main` trước khi tạo nhánh cho công việc tiếp theo. Không force-push `main`.

Đây là quy ước của nhóm trong tài liệu; chưa đồng nghĩa repo đã bật quy tắc kỹ thuật bắt buộc review hoặc chặn push trực tiếp.
