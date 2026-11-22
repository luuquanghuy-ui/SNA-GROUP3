# E3 Member 4 — FINAL Handoff

Status: **COMPLETED_REPRESENTATION_AUGMENTED_GRAPH**

## Official canonical population
- Honduras accounts: 224,685
- Labels: 0=Bad, 1=Good (used strictly for post-hoc evaluation, never for graph construction)
- Input representation: `tweet_branch` 128d embeddings extracted from `E1MultiBranch` (`e1_outputs/e1_handoff/checkpoints/best_model.pt`)

## Graph augmentation & community detection configuration
- Graph A (Baseline): 224,685 nodes, 0 directed/undirected interaction edges (inherited `BLOCKED_NO_VALIDATED_INTERACTION_EDGES` state)
- Similarity edge construction: FAISS `IndexFlatIP` cosine similarity, `top-k = 5`, `threshold >= 0.90`
- Graph B (Representation-Augmented Graph): 224,685 nodes, 995,347 undirected weighted edges (total weight normalized to 1.0)
- Community detection: Leiden algorithm (`RBConfigurationVertexPartition`, `resolution = 1.0`, `seed = 42` on active nodes; isolated nodes assigned `community_id = -1`)

## Key evaluation summary (Graph A vs. Graph B)
- Active (connected) nodes: 0 (0.00%) -> 211,733 (94.24%)
- Isolated nodes: 224,685 (100.00%) -> 12,952 (5.76%)
- Largest Connected Component (LCC): 0 -> 177,861
- Detected communities: 0 -> 2,084 (Weighted Modularity = 0.1976)
- Connected Bad accounts (out of 1,866): 0 (0.00%) -> 1,502 (80.49%) across 17 communities

## Official outputs (`final/e3_outputs/`)
- augment_graph_and_detect_communities.py
- representations/honduras_tweet_branch_128d.npy (Git LFS)
- graphs/representation_augmented_graph.graphml (Git LFS)
- communities/graph_a_communities.csv
- communities/graph_b_communities.csv
- communities/community_comparison.csv
- evaluation/graph_metrics.json
- evaluation/community_analysis.json
- figures/graph_a_visualization.png
- figures/graph_b_visualization.png
- figures/community_visualization.png

## Important notes for Member 5
- `zero_vector_rows = 177,861`: Accounts with zeroed text inputs (from E1/E2 cross-split deduplication policy) map to an identical bias vector in `tweet_branch` (cosine similarity = 1.0), forming the single baseline LCC (`177,861` nodes), while the `33,872` accounts with valid text embeddings partition into `2,083` distinct semantic communities.
- Intermediate cache files (`representations/sim_edges_*.npz`) are excluded via `.gitignore`.
