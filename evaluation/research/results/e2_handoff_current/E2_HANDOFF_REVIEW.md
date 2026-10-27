# Kiểm định bàn giao E2 mới của Thịnh

Nguồn: [50619b5cc376](https://github.com/luuquanghuy-ui/SNA-GROUP3/tree/50619b5cc37652565f14659419d8d12ca2200810). Người bàn giao xác định theo author PThinhL và nội dung E2, không theo số folder.

## Kết quả tính lại từ predictions

| method | n_accounts | positive_precision | positive_recall | positive_f1 | macro_f1 | average_precision | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train from Scratch | 44938 | 0.972393 | 0.849866 | 0.907010 | 0.953141 | 0.923331 | 317 | 9 | 56 | 44556 |
| Fox8-pretrained + Fine-tuning | 44938 | 0.990291 | 0.820375 | 0.897361 | 0.948288 | 0.917571 | 306 | 3 | 67 | 44562 |
| LR (No tweet_count) | 44938 | 0.086341 | 0.981233 | 0.158716 | 0.556605 | 0.368820 | 366 | 3873 | 7 | 40692 |

Các dòng dùng đúng 44.938 Test IDs và nhãn đối chiếu manifest (373 Bad, 44.565 Good). LR No tweet_count là kết quả tự tái lập bằng chính hàm của tác giả; xem đối chiếu trong JSON.

## Delta Transfer − Scratch

| metric | scratch | transfer | transfer_minus_scratch | difference_percentage_points |
| --- | --- | --- | --- | --- |
| positive_precision | 0.972393 | 0.990291 | 0.017899 | 1.789862 |
| positive_recall | 0.849866 | 0.820375 | -0.029491 | -2.949062 |
| positive_f1 | 0.907010 | 0.897361 | -0.009649 | -0.964931 |
| macro_f1 | 0.953141 | 0.948288 | -0.004853 | -0.485261 |
| average_precision | 0.923331 | 0.917571 | -0.005760 | -0.576025 |

Đọc chiều tăng/giảm từ delta của đúng checkpoint đang kiểm tra; không kế thừa kết luận của run cũ. Đây là điểm phần trăm, không phải phần trăm tương đối. Chưa nghiệm thu kết luận transfer cuối vì các vấn đề protocol dưới đây.

## Đối chiếu bảng do tác giả bàn giao

```json
[
  {
    "item": "scratch_metrics",
    "status": "report_mismatch",
    "discrepancies": [
      {
        "metric": "positive_precision",
        "reported": 0.996078431372549,
        "recomputed": 0.9723926380368099
      },
      {
        "metric": "positive_recall",
        "reported": 0.6809651474530831,
        "recomputed": 0.8498659517426274
      },
      {
        "metric": "positive_f1",
        "reported": 0.8089171974522293,
        "recomputed": 0.9070100143061517
      },
      {
        "metric": "macro_f1",
        "reported": 0.9037863147533646,
        "recomputed": 0.9531405634063699
      },
      {
        "metric": "average_precision",
        "reported": 0.9204162887541156,
        "recomputed": 0.9233314226627176
      }
    ]
  },
  {
    "item": "transfer_metrics",
    "status": "report_mismatch",
    "discrepancies": [
      {
        "metric": "positive_precision",
        "reported": 0.9692307692307692,
        "recomputed": 0.9902912621359223
      },
      {
        "metric": "positive_recall",
        "reported": 0.8445040214477212,
        "recomputed": 0.8203753351206434
      },
      {
        "metric": "positive_f1",
        "reported": 0.9025787965616046,
        "recomputed": 0.8973607038123167
      },
      {
        "metric": "macro_f1",
        "reported": 0.9509081383287962,
        "recomputed": 0.9482879488297182
      },
      {
        "metric": "average_precision",
        "reported": 0.91824425351955,
        "recomputed": 0.9175711723446638
      }
    ]
  }
]
```

## Kiểm tra checkpoint inference

| method | n_accounts | prediction_agreement_fraction | max_absolute_probability_difference | scores_close_rtol_1e_4_atol_1e_5 |
| --- | --- | --- | --- | --- |
| scratch | 44938 | 1.000000 | 5.410e-07 | True |
| transfer | 44938 | 1.000000 | 3.912e-07 | True |

Chỉ chạy inference, không sửa loss hay huấn luyện lại neural models. Input thực tế là TF-IDF văn bản 10.000 chiều, không phải CSV embeddings E1 128 chiều mà Diễm gửi riêng.

## Các điểm cần người phụ trách xử lý

| severity | item | finding | action |
| --- | --- | --- | --- |
| major | BCE class weighting | Targets keep Good=1, Bad=0; BCE pos_weight=119.42802450229709 multiplies Good loss, not Bad loss. | Owner must justify this objective or correct class weighting/remap explicitly and rerun both paired models; evaluation does not modify the loss. |
| major | Validation/checkpoint selection | val_df exists, but the delivered neural training function uses fixed 10 epochs without validation evaluation or validation-selected checkpoint. | Provide protocol-approved fixed-budget justification or validation checkpoint selection evidence; do not infer Test tuning solely from missing validation. |
| major | Neural seed metadata | No torch.manual_seed/np.random.seed call in delivered notebook; LR random_state=42 is not a seed for neural initialization or DataLoader shuffle. Config omits neural seed. | Owner should provide the actual run seed/environment/log or rerun both models with recorded seeds. |
| report | Label semantics | E2 report equates Bad with Bot and Good with Human; Honduras labels do not establish that equivalence. | Use Bad/Good consistently. |
| report | Overstated improvement | Current Transfer-minus-Scratch deltas: positive_precision +1.789862 pp; positive_recall -2.949062 pp; positive_f1 -0.964931 pp; macro_f1 -0.485261 pp; average_precision -0.576025 pp. One paired run is not general transfer evidence. | Report metric-specific trade-offs and percentage points; qualify conclusions by this run and protocol issues. |
| handoff | Artifact paths/export provenance | Report lists evaluation/e2_predictions_e1_emb.csv, preprocessing/scaler_e1_emb.joblib and models/model_*_e2.pt, while actual Git files are artifacts/e2_predictions.csv and artifacts/e2_*_model.pt. Export cells are absent from the received notebook. | Point handoff documentation to actual files; provide export/config generation code and any additional Drive-only artifacts. |

Bad/Good không đồng nghĩa Bot/Human. Không sửa artifact của Thịnh, không tự chọn cấu hình khác để làm đẹp kết quả. Cần đối chiếu với protocol nhóm sau khi có giải trình hoặc run sửa.
