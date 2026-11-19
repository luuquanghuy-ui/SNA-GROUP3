# E2 — Thành viên 1 · Kiểm tra Honduras và tạo tweet embedding

Script `scripts/prepare_honduras_embeddings.py` kiểm tra dữ liệu và xuất một vector MiniLM 384 chiều cho từng tài khoản Honduras. Thành viên 2 và 4 có thể đưa vector này qua `tweet_branch` của checkpoint E1. Thành viên 3 dùng hai JSONL normalized tại thư mục input và bảng metadata để đối chiếu thứ tự tài khoản.

## Dữ liệu và môi trường

- Dùng [thư mục Honduras processed](https://drive.google.com/drive/u/3/folders/1mf7CH1drH4CprPIX0sWZE23HL-thLjeC). Cần manifest, bảng split, ba file ID tương ứng, hai JSONL normalized và hai CSV audit A/B trong `quality_audit/`.
- E1 handoff lấy từ `final/e1_outputs/e1_handoff/`. Script đối chiếu chiều embedding với `feature_config.json` và ghi checksum của checkpoint. Phép mã hóa 384 chiều chạy bằng MiniLM. Checkpoint được dùng ở bước transfer hoặc chiếu 128 chiều của thành viên 2 và 4.
- Python 3.11 trở lên với NumPy, PyTorch và Transformers. Dùng bản PyTorch hỗ trợ CUDA để chạy GPU. MiniLM cần được tải từ Hugging Face ở lần đầu hoặc có sẵn trong cache.

## Chạy lại

```sh
python final/e2_outputs/scripts/prepare_honduras_embeddings.py --input-dir /path/to/processed --validate-only
python final/e2_outputs/scripts/prepare_honduras_embeddings.py --input-dir /path/to/processed --device cuda
```

Thư mục đích mặc định là `final/e2_outputs/data/`. Khi đã có output, dùng `--output-dir` để xuất lần chạy mới. Truyền lại `--revision`, `--batch-size` và chế độ precision ghi trong config để tái lập. Nếu đã tải snapshot MiniLM về máy, truyền `--model-dir` cùng `--revision` tương ứng để chạy offline. Giảm `--batch-size` nếu GPU thiếu VRAM. Tùy chọn `--fp16` chỉ dùng với CUDA và vẫn xuất mảng float32.

## Quy tắc

- Giữ đúng ID dạng chuỗi, thứ tự dòng trong manifest, split đã bàn giao và nhãn `0 = Bad`, `1 = Good`.
- Đối chiếu từng ID tweet trùng với dữ liệu normalized. Giữ bản đầu khi ID, tài khoản và nội dung sau khi bỏ khoảng trắng ở hai đầu đều khớp. Xung đột metadata vẫn có thể giữ nếu tài khoản và nội dung khớp. Nếu cùng ID tweet có tài khoản hoặc nội dung khác nhau, loại mọi bản của ID đó khỏi dữ liệu embedding.
- Trước khi lấy mẫu, loại nội dung trùng khớp chính xác sau khi chuẩn hóa khoảng trắng nếu nội dung đó có ở nhiều split. Trong mỗi tài khoản chỉ giữ lần đầu của cùng nội dung. Quy tắc này cũng loại các bản retweet có văn bản trùng giữa các split.
- Lấy tối đa năm tweet tại các vị trí cách đều theo thứ tự JSONL sau khi lọc, theo cách lấy chỉ số của E1. MiniLM giới hạn 128 token/tweet, lấy trung bình token có attention mask rồi lấy trung bình các vector tweet. Tài khoản hết tweet hợp lệ nhận vector zero để giữ nguyên ánh xạ.

## File bàn giao

| File trong `data/` | Nội dung |
| --- | --- |
| `honduras_tweet_embeddings.npy` | Mảng float32 kích thước số tài khoản × 384, quản lý bằng Git LFS |
| `honduras_labels.npy` | Nhãn int64 theo hàng |
| `honduras_split_names.npy` | `train`, `validation`, `test` theo hàng |
| `honduras_account_metadata.csv` | `row_index` bắt đầu từ 0, `account_id`, nhãn, split và số tweet |
| `honduras_embedding_config.json` | Cấu hình MiniLM, cách lọc và checksum nguồn |
| `honduras_embedding_validation.json` | Thống kê audit, kiểm tra ánh xạ, giá trị hữu hạn và checksum output |

Khi đọc bằng pandas, dùng `pd.read_csv(path, dtype={"account_id": str})`. Hàng `i` trong mọi mảng ứng với `row_index == i`.

## Kết quả bàn giao

Lượt chạy MiniLM trên RTX 3050 dùng FP16 để suy luận và xuất vector float32. Đã đối chiếu 224.685 hàng với manifest, nhãn và split. Có 157.279 tài khoản train, 22.468 validation và 44.938 test.

Sau khi loại nội dung lặp giữa các split, 89.684 tweet được chọn để mã hóa. Có 177.861 tài khoản nhận vector zero (79,16%), gồm 140/1.866 tài khoản Bad (7,50%) và 177.721/222.819 tài khoản Good (79,76%). Chênh lệch này ảnh hưởng cách diễn giải đánh giá E2 vì tình trạng thiếu văn bản có tương quan với nhãn.
