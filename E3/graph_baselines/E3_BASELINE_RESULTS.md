# E3_BASELINE_RESULTS — Người 4

## Protocol thực tế
- Scope: full Honduras corpus (224,685 accounts), theo quyết định mới của nhóm.
- Temporal window: 3600s = 1 UTC hour.
- Event size: 2–100 accounts.
- Single-signal edge threshold: shared-event weight >= 2.
- Community detection: Leiden / RBConfigurationVertexPartition.
- Resolution=1.0; seed=42; iterations=-1.
- Bad/Good labels không được dùng để tạo cạnh hoặc chọn threshold.

## Hash-only
Shared normalized hashtag + cùng UTC-hour.

## URL-only
Shared normalized URL + cùng UTC-hour.
Structured URL ưu tiên; text fallback=True và được audit.

## CoRT-only
Chỉ structured retweet target ID; không suy từ text RT @.
Status: CoRT không có structured retweet ID; Combined dùng Hash+URL.

## Combined Baseline
Signals: hash, url.
Equal coefficient = 0.500000 mỗi signal.
Normalization: max.
Công thức triển khai: tổng có trọng số bằng nhau của normalized single-signal weights.

## Metrics
| graph    |   n_nodes |   n_active_nodes |   n_edges |     density |   n_isolates |   isolated_node_rate |   largest_component_nodes |   n_communities |   largest_community_nodes |   median_community_size |   modularity |
|:---------|----------:|-----------------:|----------:|------------:|-------------:|---------------------:|--------------------------:|----------------:|--------------------------:|------------------------:|-------------:|
| hash     |    224685 |            84819 |   1905386 | 7.54861e-05 |       139866 |             0.622498 |                     79404 |            2081 |                     17517 |                       2 |     0.657811 |
| url      |    224685 |            10586 |     43953 | 1.74129e-06 |       214099 |             0.952885 |                      8678 |             391 |                      4980 |                       2 |     0.843195 |
| combined |    224685 |            86488 |   1922134 | 7.61496e-05 |       138197 |             0.61507  |                     81055 |            2127 |                     19221 |                       2 |     0.67903  |

## Interpretation / limitation
Community chỉ biểu diễn nhóm tài khoản có tín hiệu hành vi phối hợp theo protocol quan sát được; không xác nhận campaign thực tế.
Full-corpus scope là quyết định nhóm để giữ nhất quán với lần Leiden rerun; khác draft protocol trước đó từng nhắc fixed E2 Test split.

## Handoff Người 5
Dùng:
- `/content/drive/MyDrive/E3_graph_baselines/combined/edges.csv`
- `/content/drive/MyDrive/E3_graph_baselines/combined/nodes.csv`
- `/content/drive/MyDrive/E3_graph_baselines/combined/config.json`
- `/content/drive/MyDrive/E3_graph_baselines/combined/edges_with_components.csv`

Người 5 cần giữ cùng tập node và Leiden protocol khi tạo Baseline + Scratch / Baseline + Transferred.
