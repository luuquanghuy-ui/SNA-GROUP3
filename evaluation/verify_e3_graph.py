import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
E3 = ROOT / "nguoi4" / "E3_outputs-20261018T031701Z-1-001" / "E3_outputs"
OUTPUT = Path(__file__).resolve().parent / "e3_graph_recalculation.json"


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def main():
    nodes = list(rows(E3 / "e3_graph_nodes.csv"))
    position = {node["account_id"]: index for index, node in enumerate(nodes)}
    if len(position) != len(nodes):
        raise ValueError("Duplicate E3 node IDs")
    size = len(nodes)
    parent = list(range(size))
    rank = [0] * size
    degree = [0] * size
    strength = [0] * size
    communities = [node["community"] for node in nodes]
    internal_weight = defaultdict(int)
    total_weight = 0
    edge_count = 0

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        left = find(left)
        right = find(right)
        if left == right:
            return
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1

    for edge in rows(E3 / "e3_graph_edges.csv"):
        left = position[edge["source"]]
        right = position[edge["target"]]
        weight = int(edge["weight"])
        if left == right or weight < 2:
            raise ValueError("Invalid E3 edge")
        edge_count += 1
        total_weight += weight
        degree[left] += 1
        degree[right] += 1
        strength[left] += weight
        strength[right] += weight
        join(left, right)
        if communities[left] == communities[right]:
            internal_weight[communities[left]] += weight

    degree_mismatches = sum(int(node["degree"]) != degree[index] for index, node in enumerate(nodes))
    strength_mismatches = sum(int(node["strength"]) != strength[index] for index, node in enumerate(nodes))
    component_sizes = Counter(find(index) for index in range(size))
    strength_by_community = defaultdict(int)
    community_sizes = Counter()
    for index, community in enumerate(communities):
        if community != "-1":
            community_sizes[community] += 1
            strength_by_community[community] += strength[index]
    modularity = sum(
        internal_weight[community] / total_weight - (value / (2 * total_weight)) ** 2
        for community, value in strength_by_community.items()
    )
    computed = {
        "n_nodes": size,
        "n_edges": edge_count,
        "n_active_nodes": sum(value > 0 for value in degree),
        "n_isolates": sum(value == 0 for value in degree),
        "density": 2 * edge_count / (size * (size - 1)),
        "n_components": len(component_sizes),
        "largest_component_nodes": max(component_sizes.values()),
        "n_communities": len(community_sizes),
        "largest_community_nodes": max(community_sizes.values()),
        "modularity": modularity,
        "total_edge_weight": total_weight,
        "degree_mismatches": degree_mismatches,
        "strength_mismatches": strength_mismatches,
    }
    reported = next(rows(E3 / "e3_graph_metrics.csv"))
    matches = {
        name: abs(float(computed[name]) - float(reported[name])) < 1e-9
        for name in ["n_nodes", "n_edges", "n_active_nodes", "n_isolates", "density", "n_components", "largest_component_nodes", "n_communities", "largest_community_nodes", "modularity"]
    }
    if degree_mismatches or strength_mismatches:
        raise ValueError("E3 node degree/strength differs from edge list")
    audit = {
        "scope": "Independent calculation from delivered E3 node and edge CSV; does not reconstruct edges from normalized tweet records.",
        "computed": computed,
        "matches_reported": matches,
        "all_checks_pass": all(matches.values()),
    }
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    if not audit["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
