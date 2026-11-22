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

# E2 – THỰC NGHIỆM TRANSFER LEARNING TRÊN BỘ DỮ LIỆU HONDURAS

## 1. Tổng quan

### 1.1. Mục tiêu

E2 được thực hiện nhằm đánh giá khả năng chuyển giao biểu diễn đặc trưng từ mô hình đã huấn luyện trên bộ dữ liệu TwiBot-20 (E1) sang bộ dữ liệu Honduras trong bài toán phân loại tài khoản Bad/Good.

Trong E1, mô hình sử dụng kiến trúc đa nhánh để học biểu diễn tài khoản từ nhiều nguồn thông tin. Trong E2, nghiên cứu chỉ chuyển giao thành phần **Tweet Branch** đã được huấn luyện từ E1.

E2 so sánh ba cấu hình:

| Cấu hình | Mô tả |
|---|---|
| `scratch` | Khởi tạo Tweet Branch ngẫu nhiên và huấn luyện trên dữ liệu Honduras |
| `transfer` | Khởi tạo Tweet Branch bằng trọng số từ E1, sau đó fine-tune trên Honduras |
| `frozen_transfer` | Khởi tạo Tweet Branch bằng trọng số từ E1, đóng băng Tweet Branch và chỉ huấn luyện Classifier |

Mục tiêu là đánh giá liệu biểu diễn tweet được học từ TwiBot-20 có thể hỗ trợ phân loại tài khoản trên bộ dữ liệu Honduras hay không.

### 1.2. Kết quả chính

Kết quả trên Test set, tính trung bình và độ lệch chuẩn của ba random seeds:

| Cấu hình | Macro-F1 | Bad F1 | Bad AP |
|---|---:|---:|---:|
| Scratch | **0,9568 ± 0,0058** | **0,9174 ± 0,0111** | 0,9585 ± 0,0031 |
| Transfer | 0,9546 ± 0,0038 | 0,9132 ± 0,0073 | **0,9599 ± 0,0008** |
| Frozen Transfer | 0,8606 ± 0,0094 | 0,7345 ± 0,0182 | 0,7852 ± 0,0284 |

Nhận xét sơ bộ:

- Scratch đạt Macro-F1 và Bad F1 cao nhất.
- Transfer đạt Bad AP cao nhất.
- Frozen Transfer có hiệu suất thấp hơn đáng kể so với hai cấu hình còn lại.
- Việc chuyển giao Tweet Branch chưa tạo ra cải thiện rõ ràng về Macro-F1 và Bad F1 so với huấn luyện từ đầu trong thiết lập thực nghiệm hiện tại.

## 2. Dữ liệu thực nghiệm

### 2.1. Nguồn dữ liệu và đặc trưng đầu vào

E2 sử dụng dữ liệu Honduras đã được chuẩn bị và tiền xử lý trước đó.

| Thuộc tính | Giá trị |
|---|---|
| Bộ dữ liệu | Honduras |
| Đặc trưng đầu vào | Tweet Embedding |
| Kích thước embedding | 384 chiều |
| Mô hình embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Nhãn 0 | Bad |
| Nhãn 1 | Good |

### 2.2. Lựa chọn cohort thực nghiệm

Tổng số tài khoản ban đầu trong bộ dữ liệu Honduras là 224.685.

Qua kiểm tra dữ liệu, có 79,16% tài khoản không có tweet được chọn để tạo embedding, dẫn đến vector embedding bằng 0.

Để tránh đưa các tài khoản không có thông tin tweet vào thực nghiệm chính, E2 chỉ sử dụng những tài khoản thỏa mãn điều kiện:

`selected_tweet_count > 0`

Sau khi lọc, tập dữ liệu thực nghiệm gồm 46.824 tài khoản.

| Nhãn | Số lượng | Tỷ lệ |
|---|---:|---:|
| Bad | 1.726 | 3,69% |
| Good | 45.098 | 96,31% |
| **Tổng** | **46.824** | **100%** |

**Lưu ý quan trọng:** Kết quả E2 chỉ áp dụng cho 46.824 tài khoản có ít nhất một tweet được chọn, không đại diện trực tiếp cho toàn bộ 224.685 tài khoản Honduras.

### 2.3. Chia tập dữ liệu

E2 sử dụng account-level Train/Validation/Test split đã được Member 1 chuẩn bị từ trước. Không thực hiện chia ngẫu nhiên lại dữ liệu.

| Tập dữ liệu | Bad | Good | Tổng |
|---|---:|---:|---:|
| Train | 1.209 | 34.246 | 35.455 |
| Validation | 172 | 3.469 | 3.641 |
| Test | 345 | 7.383 | 7.728 |
| **Tổng** | **1.726** | **45.098** | **46.824** |

Mục đích của từng tập:

- **Train:** Dùng để huấn luyện và cập nhật tham số mô hình.
- **Validation:** Dùng để đánh giá trong quá trình huấn luyện, lựa chọn checkpoint tốt nhất và thực hiện early stopping.
- **Test:** Chỉ dùng để đánh giá cuối cùng khả năng tổng quát hóa của mô hình.

Các thông tin provenance trong manifest không được sử dụng làm đặc trưng đầu vào nhằm hạn chế nguy cơ rò rỉ nhãn.

## 3. Kiến trúc mô hình

### 3.1. Kiến trúc E2

E2 sử dụng kiến trúc một nhánh, tập trung vào biểu diễn tweet:

`Tweet Embedding (384) → Tweet Branch (128) → Classifier (2)`

Chi tiết các thành phần:

| Thành phần | Cấu hình | Chức năng |
|---|---|---|
| Linear | 384 → 128 | Chiếu embedding về không gian 128 chiều |
| LayerNorm | 128 | Chuẩn hóa biểu diễn |
| GELU | Activation | Tăng khả năng biểu diễn phi tuyến |
| Dropout | 0,3 | Hạn chế overfitting |
| Classifier | 128 → 2 | Tạo logits cho hai lớp Bad và Good |

### 3.2. Trọng số chuyển giao từ E1

Checkpoint nguồn: `e1/best_model.pt`

Chỉ thành phần `tweet_branch` được chuyển giao từ E1 sang E2.

Các tham số được sử dụng:

| Tham số | Kích thước |
|---|---|
| `tweet_branch.0.weight` | (128, 384) |
| `tweet_branch.0.bias` | (128,) |
| `tweet_branch.1.weight` | (128,) |
| `tweet_branch.1.bias` | (128,) |

Các thành phần sau không được chuyển giao:

- Fusion module.
- Các nhánh đặc trưng khác.
- Classifier của E1.
- Projector của E1.

SHA256 của checkpoint E1 đã xác minh:

`219c21ca894c6ea4bcd8a4ac009f0383880892aa0df665795e0ef8f880ce7394`

### 3.3. So sánh ba cấu hình

| Cấu hình | Khởi tạo Tweet Branch | Tweet Branch có cập nhật? | Classifier |
|---|---|---|---|
| Scratch | Ngẫu nhiên | Có | Khởi tạo mới |
| Transfer | Trọng số từ E1 | Có | Khởi tạo mới |
| Frozen Transfer | Trọng số từ E1 | Không | Khởi tạo mới |

Ý nghĩa:

- **Scratch:** Đóng vai trò baseline để đánh giá hiệu quả của việc huấn luyện từ đầu.
- **Transfer:** Kiểm tra khả năng tận dụng và điều chỉnh biểu diễn tweet đã học từ E1.
- **Frozen Transfer:** Kiểm tra khả năng sử dụng trực tiếp biểu diễn từ E1 mà không cập nhật Tweet Branch.

## 4. Thiết kế và thiết lập thực nghiệm

### 4.1. Cấu hình huấn luyện

Ba cấu hình sử dụng cùng thiết lập huấn luyện nhằm bảo đảm khả năng so sánh.

| Tham số | Giá trị |
|---|---|
| Input dimension | 384 |
| Tweet Branch output dimension | 128 |
| Dropout | 0,3 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 0,0002 |
| Weight decay | 0,01 |
| Maximum epochs | 30 |
| Early stopping patience | 5 |
| Loss function | Weighted Cross Entropy |
| Random seeds | 42, 52, 62 |

### 4.2. Class Weights

Dữ liệu có sự mất cân bằng đáng kể giữa hai lớp. Lớp Bad chỉ chiếm khoảng 3,69% cohort thực nghiệm.

Để tăng mức độ chú ý của mô hình đối với lớp thiểu số, E2 sử dụng Weighted Cross Entropy.

Trọng số lớp được tính từ tập Train theo công thức:

`weight_c = sqrt(N / (2 × class_count_c))`

Trong đó:

- `N`: Tổng số mẫu trong tập Train.
- `class_count_c`: Số lượng mẫu thuộc lớp c.

Trọng số được tính một lần từ Train và sử dụng thống nhất cho tất cả cấu hình và seeds.

| Lớp | Class weight |
|---|---:|
| Bad (0) | 3,8292224 |
| Good (1) | 0,71948016 |

Không kết hợp Weighted Random Sampler với Weighted Cross Entropy.

### 4.3. Random Seeds

Mỗi cấu hình được huấn luyện với ba random seeds:

- Seed 42.
- Seed 52.
- Seed 62.

Tổng số lần huấn luyện:

`3 configurations × 3 seeds = 9 training runs`

Việc sử dụng nhiều seeds giúp đánh giá mức độ biến động của kết quả khi thay đổi khởi tạo ngẫu nhiên.

### 4.4. Lựa chọn checkpoint

Sau mỗi epoch, mô hình được đánh giá trên Validation set.

Quy tắc lựa chọn checkpoint:

1. Ưu tiên checkpoint có Validation Macro-F1 cao nhất.
2. Nếu Macro-F1 bằng nhau, ưu tiên checkpoint có Validation Loss thấp hơn.
3. Áp dụng early stopping với patience bằng 5 epoch.

Test set không được sử dụng để lựa chọn checkpoint hoặc điều chỉnh tham số mô hình.

## 5. Các chỉ số đánh giá

E2 sử dụng các chỉ số sau:

| Chỉ số | Ý nghĩa |
|---|---|
| Accuracy | Tỷ lệ dự đoán đúng trên toàn bộ mẫu |
| Macro Precision | Precision trung bình của hai lớp |
| Macro Recall | Recall trung bình của hai lớp |
| Macro-F1 | F1 trung bình của hai lớp |
| Bad Precision | Precision của lớp Bad (0) |
| Bad Recall | Recall của lớp Bad (0) |
| Bad F1 | F1-score của lớp Bad (0) |
| Bad AP | Average Precision của lớp Bad (0) |

**Lưu ý về Bad AP:** Vì Bad được mã hóa là nhãn 0, xác suất dùng để tính Average Precision là:

`probabilities[:, 0]`

Các kết quả tổng hợp được trình bày dưới dạng:

`Mean ± Sample Standard Deviation`

Độ lệch chuẩn được tính bằng `ddof = 1` trên ba seeds.

## 6. Kết quả thực nghiệm

### 6.1. Kết quả Validation

Bảng dưới đây trình bày kết quả trung bình và độ lệch chuẩn của ba seeds trên Validation set.

| Cấu hình | Accuracy | Macro-F1 | Bad Precision | Bad Recall | Bad F1 | Bad AP |
|---|---:|---:|---:|---:|---:|---:|
| Scratch | 0,9939 ± 0,0006 | **0,9655 ± 0,0029** | 0,9484 ± 0,0168 | 0,9205 ± 0,0089 | **0,9342 ± 0,0055** | 0,9623 ± 0,0012 |
| Transfer | 0,9936 ± 0,0006 | 0,9641 ± 0,0036 | 0,9408 ± 0,0096 | **0,9225 ± 0,0121** | 0,9315 ± 0,0069 | **0,9629 ± 0,0013** |
| Frozen Transfer | 0,9729 ± 0,0004 | 0,8628 ± 0,0026 | 0,6769 ± 0,0058 | 0,8159 ± 0,0146 | 0,7399 ± 0,0050 | 0,7909 ± 0,0291 |

### 6.2. Kết quả Test

Bảng dưới đây trình bày kết quả đánh giá cuối cùng trên Test set.

| Cấu hình | Accuracy | Macro-F1 | Bad Precision | Bad Recall | Bad F1 | Bad AP |
|---|---:|---:|---:|---:|---:|---:|
| Scratch | **0,9927 ± 0,0011** | **0,9568 ± 0,0058** | **0,9304 ± 0,0303** | **0,9053 ± 0,0073** | **0,9174 ± 0,0111** | 0,9585 ± 0,0031 |
| Transfer | 0,9924 ± 0,0007 | 0,9546 ± 0,0038 | 0,9284 ± 0,0170 | 0,8986 ± 0,0029 | 0,9132 ± 0,0073 | **0,9599 ± 0,0008** |
| Frozen Transfer | 0,9747 ± 0,0013 | 0,8606 ± 0,0094 | 0,6901 ± 0,0073 | 0,7855 ± 0,0342 | 0,7345 ± 0,0182 | 0,7852 ± 0,0284 |

### 6.3. Confusion Matrix trên Test set

Thứ tự nhãn trong các ma trận:

- Hàng: Actual Label `[Bad, Good]`.
- Cột: Predicted Label `[Bad, Good]`.

#### Scratch

| Seed | Confusion Matrix |
|---|---|
| 42 | `[[310, 35], [12, 7371]]` |
| 52 | `[[312, 33], [25, 7358]]` |
| 62 | `[[315, 30], [34, 7349]]` |

#### Transfer

| Seed | Confusion Matrix |
|---|---|
| 42 | `[[310, 35], [19, 7364]]` |
| 52 | `[[311, 34], [31, 7352]]` |
| 62 | `[[309, 36], [22, 7361]]` |

#### Frozen Transfer

| Seed | Confusion Matrix |
|---|---|
| 42 | `[[258, 87], [120, 7263]]` |
| 52 | `[[274, 71], [119, 7264]]` |
| 62 | `[[281, 64], [126, 7257]]` |

Trong đó:

- Ô `[0,0]`: Actual Bad, Predicted Bad.
- Ô `[0,1]`: Actual Bad, Predicted Good.
- Ô `[1,0]`: Actual Good, Predicted Bad.
- Ô `[1,1]`: Actual Good, Predicted Good.

Chín biểu đồ Confusion Matrix được lưu riêng dưới dạng PNG trong thư mục `results/confusion_matrices/`.

## 7. Phân tích kết quả

### 7.1. Scratch và Transfer

Trên Test set, Scratch đạt Macro-F1 trung bình 0,9568, cao hơn Transfer với 0,9546.

Bad F1 của Scratch đạt 0,9174, trong khi Transfer đạt 0,9132.

Hai cấu hình có kết quả tương đối gần nhau. Điều này cho thấy việc chuyển giao Tweet Branch từ TwiBot-20 và fine-tune trên Honduras chưa tạo ra mức cải thiện rõ ràng so với huấn luyện từ đầu.

Tuy nhiên, Transfer đạt Bad AP cao hơn Scratch (0,9599 so với 0,9585). Điều này cho thấy Transfer có kết quả nhỉnh hơn về khả năng xếp hạng xác suất cho lớp Bad theo Average Precision, mặc dù chưa vượt Scratch về F1.

### 7.2. Frozen Transfer

Frozen Transfer có Macro-F1 trung bình 0,8606 và Bad F1 trung bình 0,7345 trên Test set.

Kết quả thấp hơn đáng kể so với Scratch và Transfer. Điều này cho thấy việc đóng băng hoàn toàn Tweet Branch có thể hạn chế khả năng thích nghi của biểu diễn với đặc điểm dữ liệu Honduras.

Cấu hình Transfer cho phép cập nhật Tweet Branch đạt kết quả tốt hơn Frozen Transfer, cho thấy việc fine-tune nhánh đặc trưng có vai trò quan trọng trong thực nghiệm hiện tại.

### 7.3. Ảnh hưởng của mất cân bằng lớp

Lớp Bad chỉ chiếm khoảng 3,69% cohort thực nghiệm. Vì vậy, Accuracy cao không đồng nghĩa với việc mô hình nhận diện tốt lớp Bad.

E2 sử dụng Weighted Cross Entropy và theo dõi thêm Macro-F1, Bad Precision, Bad Recall, Bad F1 và Bad AP.

Các Confusion Matrix cho thấy Scratch và Transfer nhận diện được phần lớn tài khoản Bad, trong khi Frozen Transfer bỏ sót nhiều tài khoản Bad hơn.

### 7.4. Độ ổn định giữa các seeds

Kết quả Test Macro-F1:

| Cấu hình | Mean ± STD |
|---|---:|
| Scratch | 0,9568 ± 0,0058 |
| Transfer | 0,9546 ± 0,0038 |
| Frozen Transfer | 0,8606 ± 0,0094 |

Các kết quả được tổng hợp từ ba seeds nhằm hạn chế sự phụ thuộc vào một lần khởi tạo ngẫu nhiên.

Tuy nhiên, ba seeds chỉ cung cấp đánh giá ban đầu về độ biến động, chưa đủ để khẳng định sự khác biệt giữa các cấu hình có ý nghĩa thống kê.

### 7.5. Kết luận từ kết quả

Trong thiết lập E2 hiện tại:

- Scratch là cấu hình tốt nhất về Test Macro-F1 và Bad F1.
- Transfer có Test Bad AP cao nhất.
- Frozen Transfer có hiệu suất thấp nhất.
- Fine-tune Tweet Branch có hiệu quả hơn việc đóng băng hoàn toàn nhánh này.
- Chưa có bằng chứng cho thấy chuyển giao Tweet Branch giúp cải thiện rõ ràng Macro-F1 hoặc Bad F1 so với huấn luyện từ đầu.

## 8. Hạn chế của thực nghiệm

1. **Phạm vi dữ liệu:** E2 chỉ sử dụng 46.824 tài khoản có ít nhất một tweet được chọn, không đại diện cho toàn bộ 224.685 tài khoản ban đầu.
2. **Mất cân bằng lớp:** Lớp Bad chỉ chiếm 3,69%, gây khó khăn cho việc nhận diện lớp thiểu số.
3. **Phạm vi chuyển giao:** Chỉ Tweet Branch của E1 được chuyển giao, không sử dụng các nhánh đặc trưng khác và Fusion module.
4. **Số lượng seeds:** Thực nghiệm sử dụng ba seeds, chưa đủ để đưa ra kết luận thống kê mạnh.
5. **Phạm vi đặc trưng:** E2 chỉ sử dụng Tweet Embedding, chưa khai thác đầy đủ thông tin hồ sơ tài khoản và cấu trúc mạng tương tác.
6. **Phạm vi kết luận:** Phân loại Bad/Good không trực tiếp chứng minh một tài khoản tham gia chiến dịch phối hợp. Cần kết hợp với phân tích mạng và bằng chứng hành vi trong các thực nghiệm tiếp theo.

## 9. Cấu trúc và ý nghĩa các file đầu ra

Thư mục kết quả chính:

`/results`

Các checkpoint được lưu tại thư mục `E2_CHECKPOINT_DIR` trong notebook. Cần sử dụng đúng đường dẫn biến này khi truy xuất checkpoint.

### 9.1. Model Checkpoints

| File | Số lượng | Kết quả lưu trữ | Mục đích sử dụng |
|---|---:|---|---|
| `scratch_seed42_best.pt` | 1 | Scratch, seed 42 | Lưu mô hình Scratch tốt nhất |
| `scratch_seed52_best.pt` | 1 | Scratch, seed 52 | Lưu mô hình Scratch tốt nhất |
| `scratch_seed62_best.pt` | 1 | Scratch, seed 62 | Lưu mô hình Scratch tốt nhất |
| `transfer_seed42_best.pt` | 1 | Transfer, seed 42 | Lưu mô hình Transfer tốt nhất |
| `transfer_seed52_best.pt` | 1 | Transfer, seed 52 | Lưu mô hình Transfer tốt nhất |
| `transfer_seed62_best.pt` | 1 | Transfer, seed 62 | Lưu mô hình Transfer tốt nhất |
| `frozen_transfer_seed42_best.pt` | 1 | Frozen Transfer, seed 42 | Lưu mô hình Frozen Transfer tốt nhất |
| `frozen_transfer_seed52_best.pt` | 1 | Frozen Transfer, seed 52 | Lưu mô hình Frozen Transfer tốt nhất |
| `frozen_transfer_seed62_best.pt` | 1 | Frozen Transfer, seed 62 | Lưu mô hình Frozen Transfer tốt nhất |

**Tổng cộng: 9 checkpoint.**

Các checkpoint lưu trọng số của mô hình tại epoch tốt nhất, được lựa chọn theo Validation Macro-F1 và quy tắc tie-break Validation Loss.

### 9.2. Training History

| File pattern | Số lượng | Kết quả lưu trữ | Mục đích sử dụng |
|---|---:|---|---|
| `scratch_seed*_history.json` | 3 | Lịch sử huấn luyện Scratch | Phân tích quá trình học qua các epoch |
| `transfer_seed*_history.json` | 3 | Lịch sử huấn luyện Transfer | Phân tích quá trình fine-tune |
| `frozen_transfer_seed*_history.json` | 3 | Lịch sử huấn luyện Frozen Transfer | Phân tích quá trình huấn luyện Classifier |

**Tổng cộng: 9 history JSON.**

Các file này lưu lịch sử huấn luyện và đánh giá theo epoch, phục vụ việc theo dõi sự thay đổi của loss và các chỉ số trong quá trình huấn luyện.

### 9.3. Evaluation Results

| File | Kết quả lưu trữ | Mục đích sử dụng |
|---|---|---|
| `validation_results_by_seed.csv` | Kết quả Validation của từng cấu hình và seed | So sánh hiệu suất từng lần chạy trên Validation |
| `test_results_by_seed.csv` | Kết quả Test của từng cấu hình và seed | So sánh hiệu suất từng lần chạy trên Test |
| `validation_mean_std.csv` | Mean và STD của Validation | Tổng hợp kết quả Validation trên ba seeds |
| `test_mean_std.csv` | Mean và STD của Test | Tổng hợp kết quả Test trên ba seeds |

Đây là bốn file quan trọng nhất để tổng hợp kết quả thực nghiệm và viết báo cáo.

### 9.4. Confusion Matrix PNG

Thư mục:

`results/confusion_matrices/`

| File pattern | Số lượng | Kết quả lưu trữ | Mục đích sử dụng |
|---|---:|---|---|
| `scratch_seed*_confusion_matrix.png` | 3 | Confusion Matrix của Scratch | Trực quan hóa kết quả phân loại Scratch |
| `transfer_seed*_confusion_matrix.png` | 3 | Confusion Matrix của Transfer | Trực quan hóa kết quả phân loại Transfer |
| `frozen_transfer_seed*_confusion_matrix.png` | 3 | Confusion Matrix của Frozen Transfer | Trực quan hóa kết quả phân loại Frozen Transfer |

**Tổng cộng: 9 hình PNG.**

Mỗi hình biểu diễn một ma trận 2×2, bao gồm số lượng dự đoán và tỷ lệ phần trăm theo từng hàng (Actual Label).

Các hình có thể được sử dụng trực tiếp trong báo cáo.

### 9.5. Confusion Matrix CSV

Thư mục:

`results/confusion_matrices_csv/`

| File pattern | Số lượng | Kết quả lưu trữ | Mục đích sử dụng |
|---|---:|---|---|
| `scratch_seed*_confusion_matrix.csv` | 3 | Số lượng dự đoán của Scratch | Lưu dữ liệu ma trận gốc |
| `transfer_seed*_confusion_matrix.csv` | 3 | Số lượng dự đoán của Transfer | Lưu dữ liệu ma trận gốc |
| `frozen_transfer_seed*_confusion_matrix.csv` | 3 | Số lượng dự đoán của Frozen Transfer | Lưu dữ liệu ma trận gốc |

**Tổng cộng: 9 file CSV.**

Các file chứa số lượng mẫu thuộc từng ô của Confusion Matrix, có thể sử dụng để phân tích hoặc tạo lại biểu đồ.

### 9.6. Experiment Configuration và Audit

| File | Kết quả lưu trữ | Mục đích sử dụng |
|---|---|---|
| `e2_experiment_config.json` | Cấu hình dữ liệu, kiến trúc, nguồn chuyển giao, hyperparameters, seeds, class weights và metric | Ghi lại thiết lập thực nghiệm để tái sử dụng và kiểm tra |
| `e2_artifact_audit.csv` | Danh sách artifact, trạng thái tồn tại, dung lượng và đường dẫn | Kiểm tra tính đầy đủ của các file đầu ra |

File JSON là tài liệu mô tả cấu hình thực nghiệm. File audit là kết quả kiểm tra các artifact sau khi hoàn thành E2.

## 10. Tổng hợp Artifact và trạng thái hoàn thành

Cell Artifact Audit đã kiểm tra 36 file thuộc bốn nhóm:

| Loại artifact | Số lượng cần có | Số lượng tìm thấy | Trạng thái |
|---|---:|---:|---|
| Model checkpoints | 9 | 9 | Hoàn thành |
| Training histories | 9 | 9 | Hoàn thành |
| Confusion Matrix PNG | 9 | 9 | Hoàn thành |
| Confusion Matrix CSV | 9 | 9 | Hoàn thành |
| **Tổng cộng** | **36** | **36** | **PASS** |

Kết quả kiểm tra:

```text
Total artifacts checked: 36
Existing artifacts: 36
Missing artifacts: 0

SUCCESS: All E2 artifacts are available.
```

Ngoài 36 artifact trên, thư mục results còn có các bảng kết quả CSV, file cấu hình JSON và báo cáo audit.

## 11. Quy trình sử dụng và tiếp tục thực nghiệm

### 11.1. Truy xuất kết quả

Có thể sử dụng trực tiếp các file CSV để đọc kết quả:

- Đọc `validation_results_by_seed.csv` để xem kết quả từng lần chạy trên Validation.
- Đọc `test_results_by_seed.csv` để xem kết quả từng lần chạy trên Test.
- Đọc `validation_mean_std.csv` để xem kết quả tổng hợp Validation.
- Đọc `test_mean_std.csv` để xem kết quả tổng hợp Test.

### 11.2. Truy xuất checkpoint

Sử dụng đúng đường dẫn checkpoint đã được cấu hình trong notebook:

`E2_CHECKPOINT_DIR`

Tên file có định dạng:

`{configuration}_seed{seed}_best.pt`

Ví dụ:

`transfer_seed42_best.pt`

Checkpoint có thể được nạp lại để đánh giá hoặc sử dụng trong các phân tích tiếp theo mà không cần huấn luyện lại.

### 11.3. Sử dụng Confusion Matrix

- Sử dụng PNG khi cần xem hoặc đưa biểu đồ vào báo cáo.
- Sử dụng CSV khi cần đọc số lượng dự đoán hoặc tạo lại biểu đồ.

Không cần chạy lại quá trình huấn luyện chỉ để lấy các ma trận đã được lưu.

## 12. Kết luận và bàn giao E2

E2 đã hoàn thành các công việc:

- Chuẩn bị cohort thực nghiệm Honduras.
- Sử dụng account-level Train/Validation/Test split.
- Xây dựng ba cấu hình Scratch, Transfer và Frozen Transfer.
- Chuyển giao Tweet Branch từ E1 cho hai cấu hình Transfer.
- Huấn luyện với ba random seeds: 42, 52 và 62.
- Đánh giá trên Validation và Test.
- Tổng hợp Mean ± Standard Deviation.
- Tạo chín Confusion Matrix PNG và chín Confusion Matrix CSV.
- Lưu chín checkpoint và chín training history.
- Xuất file cấu hình thực nghiệm và báo cáo Artifact Audit.
- Kiểm tra thành công 36/36 artifact.

**Trạng thái: Hoàn thành phần triển khai và đánh giá thực nghiệm E2.**

Các kết quả E2 có thể được sử dụng để tổng hợp vào báo cáo chung của nhóm và làm cơ sở cho các phân tích tiếp theo kết hợp thông tin tài khoản, đặc trưng tweet và cấu trúc mạng tương tác.

Lưu ý cuối cùng: E2 thực hiện phân loại tài khoản Bad/Good, không trực tiếp xác định tài khoản có tham gia chiến dịch phối hợp hay không. Kết luận về coordinated campaigns cần dựa trên các phân tích bổ sung ở những phần nghiên cứu khác.