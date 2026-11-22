# E3 Member 3 — FINAL Handoff

Status: **BLOCKED_NO_VALIDATED_INTERACTION_EDGES**

## Official canonical population
- Honduras accounts: 224,685
- Labels: 0=Bad, 1=Good
- Splits: train / validation / test
- Labels are NOT used to create graph edges.

## Structured interaction data

- Canonical structured directed edges: 0

## Neighbor data provenance
- target accounts: 11,826
- target overlap with Honduras: 93
- target overlap ratio: 0.0079
- split agreement: 0.4839
- auto merge allowed: False
- merged into official graph: False

## Official outputs
- data/nodes.csv
- data/interaction_edges.csv
- data/interaction_edges_structured.csv
- data/interaction_events.csv.gz
- data/tweet_id_user_mapping.csv
- data/neighbor_edges_honduras_overlap.csv
- data/neighbor_edges_expanded.csv
- data/graph_data_report.json
- audit/data_readiness.json
- graphs/interaction_graph_directed.graphml
- graphs/coordination_graph.graphml
- figures/interaction_graph.png (official graph, nếu có edge)
- figures/auxiliary_neighbor_overlap.png (chỉ dùng khi official graph rỗng),
## Important limitations
- Do not infer retweet from text such as `RT @username`.
- Neighbor data is not treated as Honduras unless provenance audit passes.
- External support accounts do not receive Honduras labels/splits.
- Zero text embedding accounts remain canonical nodes; embedding availability is a Member-4 concern.