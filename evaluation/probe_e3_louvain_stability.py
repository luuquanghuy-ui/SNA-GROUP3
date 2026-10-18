import csv
import json
import os
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "nguoi4" / "E3_outputs-20261018T031701Z-1-001" / "E3_outputs"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


graph = nx.Graph()
graph.add_nodes_from(sorted(row["account_id"] for row in read_rows(SOURCE / "e3_graph_nodes.csv")))
graph.add_weighted_edges_from(
    (row["source"], row["target"], int(row["weight"]))
    for row in read_rows(SOURCE / "e3_graph_edges.csv")
)
active = graph.subgraph(sorted(node for node, degree in graph.degree() if degree > 0)).copy()
communities = nx.community.louvain_communities(active, weight="weight", resolution=1, seed=42)
communities = sorted(communities, key=lambda group: (-len(group), min(group)))
result = {
    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    "networkx": nx.__version__,
    "n_communities": len(communities),
    "largest_community_nodes": len(communities[0]),
    "modularity": nx.community.modularity(active, communities, weight="weight", resolution=1),
}
out = Path(__file__).resolve().parent / "e3_rerun_from_tweets" / f"louvain_hash_{result['pythonhashseed']}.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
