#!/usr/bin/env python3
"""
E3 (Member 4): Build Representation-Augmented Graph & Community Analysis
Môi trường: Local / WSL (Jupyter Notebook hoặc CLI)

Chức năng chính:
1. Nạp checkpoint E1 (best_model.pt) và khởi tạo kiến trúc E1MultiBranch từ e1_multibranch.py
   (tự động dùng lại honduras_tweet_branch_128d.npy nếu đã tồn tại).
2. Tính Cosine Similarity (top-k = 5, threshold >= 0.9) bằng FAISS trên toàn bộ 224.685 tài khoản
   (tự động lưu/nạp cache .npz để tăng tốc các lần chạy sau).
3. Đọc Graph A từ interaction_edges.csv (cộng tổng 2 chiều); xử lý an toàn cả trường hợp 0 cạnh.
4. Trộn Graph A và Similarity Graph (lambda = 0.5) tạo Graph B và xuất GraphML tối ưu bộ nhớ qua igraph.
5. Phát hiện cộng đồng Leiden (resolution = 1.0, seed = 42) cho Graph A, Graph B và Equal-Edge Control (nếu M > 0).
6. Đánh giá cấu trúc đồ thị (ARI, Modularity, LCC, Isolate rate) và phân tích hậu nghiệm nhãn Bad/Good (0 = Bad, 1 = Good).
7. Xuất toàn bộ bảng đánh giá CSV/JSON và biểu đồ trực quan hóa.
"""

import gc
import json
import os
import sys
from pathlib import Path

import faiss
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN & SIÊU THAM SỐ
# ==============================================================================
BASE_DIR = Path(os.environ.get("SNA_BASE_DIR", Path(__file__).resolve().parent))

PATHS = {
    "e1_model_dir": BASE_DIR / "e1_outputs" / "e1_handoff" / "model",
    "e1_ckpt": BASE_DIR / "e1_outputs" / "e1_handoff" / "checkpoints" / "best_model.pt",
    "e2_emb": BASE_DIR / "e2_outputs" / "data" / "honduras_tweet_embeddings.npy",
    "e2_meta": BASE_DIR / "e2_outputs" / "data" / "honduras_account_metadata.csv",
    "e3_nodes": BASE_DIR / "e3_outputs" / "data" / "nodes.csv",
    "e3_edges": BASE_DIR / "e3_outputs" / "data" / "interaction_edges.csv",
    "e3_aux_edges": BASE_DIR / "e3_outputs" / "data" / "neighbor_edges_honduras_overlap.csv",
    "out_dir": BASE_DIR / "e3_outputs",
}

CONFIG = {
    "top_k": 5,
    "sim_threshold": 0.90,
    "lambda_blend": 0.50,
    "leiden_resolution": 1.0,
    "leiden_seed": 42,
    "leiden_iterations": -1,
    "batch_size": 4096,
    "run_equal_edge_control": True,
    "use_cached_128d": True,
    "use_cached_sim_edges": True,
    "use_auxiliary_neighbor_if_empty": False,
}


def setup_directories(out_dir: Path) -> None:
    for sub in ["representations", "graphs", "communities", "evaluation", "figures"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def extract_tweet_branch_representations(
    model_dir: Path,
    ckpt_path: Path,
    emb_path: Path,
    out_rep_path: Path,
    batch_size: int = 4096,
    use_cache: bool = True,
) -> np.ndarray:
    """Bước 1-3: Load E1MultiBranch, trích xuất riêng tweet_branch và sinh vector 128d."""
    if use_cache and out_rep_path.exists():
        reps_128d = np.load(out_rep_path).astype(np.float32)
        if reps_128d.ndim == 2 and reps_128d.shape[1] == 128 and not np.isnan(reps_128d).any():
            print(f"[1/6] Đã nạp biểu diễn 128d từ cache ({reps_128d.shape}): {out_rep_path}")
            return reps_128d

    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from e1_multibranch import E1MultiBranch  # noqa: E402

    embeddings_in = np.load(emb_path)
    assert embeddings_in.ndim == 2 and embeddings_in.shape[1] == 384, (
        f"Kỳ vọng embedding đầu vào 384 chiều, nhận được {embeddings_in.shape}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/6] Thiết bị tính toán: {device} | Input shape: {embeddings_in.shape}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    tweet_dim = state_dict["tweet_branch.0.weight"].shape[1]
    branch_dim = state_dict["tweet_branch.0.weight"].shape[0]
    profile_text_dim = state_dict["profile_text_branch.0.weight"].shape[1]
    profile_struct_dim = state_dict["profile_struct_branch.0.weight"].shape[1]
    neighbor_dim = state_dict["neighbor_branch.0.weight"].shape[1]
    fusion_dim = state_dict["fusion.0.weight"].shape[0]
    projection_dim = state_dict["projector.2.weight"].shape[0]
    num_classes = state_dict["classifier.weight"].shape[0]

    model = E1MultiBranch(
        tweet_dim=tweet_dim,
        profile_text_dim=profile_text_dim,
        profile_struct_dim=profile_struct_dim,
        neighbor_dim=neighbor_dim,
        branch_dim=branch_dim,
        fusion_dim=fusion_dim,
        projection_dim=projection_dim,
        num_classes=num_classes,
    )
    model.load_state_dict(state_dict, strict=True)

    tweet_branch = model.tweet_branch.to(device)
    tweet_branch.eval()

    reps_list = []
    with torch.no_grad():
        for start in range(0, len(embeddings_in), batch_size):
            batch_x = torch.tensor(
                embeddings_in[start : start + batch_size], dtype=torch.float32, device=device
            )
            reps_list.append(tweet_branch(batch_x).cpu().numpy())

    reps_128d = np.vstack(reps_list).astype(np.float32)
    assert not np.isnan(reps_128d).any() and not np.isinf(reps_128d).any(), (
        "Phát hiện giá trị NaN hoặc Inf trong biểu diễn 128d!"
    )

    np.save(out_rep_path, reps_128d)
    print(f"      -> Đã lưu biểu diễn 128d ({reps_128d.shape}) tại: {out_rep_path}")

    del embeddings_in, model, tweet_branch, checkpoint, state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return reps_128d


def build_similarity_edges(
    reps_128d: np.ndarray,
    cache_path: Path,
    top_k: int = 5,
    threshold: float = 0.90,
    use_cache: bool = True,
) -> dict[tuple[int, int], float]:
    """Bước 4-5: Tính Cosine Similarity bằng FAISS và lọc cạnh top-k >= threshold."""
    if use_cache and cache_path.exists():
        data = np.load(cache_path)
        u_arr, v_arr, w_arr = data["u"], data["v"], data["w"]
        sim_edge_dict = {
            (int(u), int(v)): float(w) for u, v, w in zip(u_arr, v_arr, w_arr)
        }
        print(f"[2/6] Đã nạp {len(sim_edge_dict):,} cạnh Similarity từ cache: {cache_path}")
        return sim_edge_dict

    print(f"[2/6] Tính Cosine Similarity (top-k={top_k}, threshold>={threshold})...")
    norms = np.linalg.norm(reps_128d, axis=1, keepdims=True)
    valid_mask = norms.squeeze() > 1e-12

    reps_norm = np.zeros_like(reps_128d, dtype=np.float32)
    reps_norm[valid_mask] = reps_128d[valid_mask] / norms[valid_mask]

    dim = reps_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(reps_norm)

    similarities, indices = index.search(reps_norm, top_k + 1)

    sim_edge_dict: dict[tuple[int, int], float] = {}
    num_nodes = len(reps_128d)
    for src_idx in range(num_nodes):
        if not valid_mask[src_idx]:
            continue
        for rank in range(1, top_k + 1):
            dst_idx = int(indices[src_idx, rank])
            score = float(similarities[src_idx, rank])
            if dst_idx < 0 or dst_idx == src_idx or not valid_mask[dst_idx]:
                continue
            if score >= threshold:
                u, v = (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)
                if (u, v) not in sim_edge_dict or score > sim_edge_dict[(u, v)]:
                    sim_edge_dict[(u, v)] = score

    print(f"      -> Số cạnh Similarity vô hướng: {len(sim_edge_dict):,}")

    # Lưu cache để các lần chạy sau chỉ mất < 1 giây
    if len(sim_edge_dict) > 0:
        edges_arr = np.array(list(sim_edge_dict.keys()), dtype=np.int32)
        weights_arr = np.array(list(sim_edge_dict.values()), dtype=np.float32)
        np.savez_compressed(cache_path, u=edges_arr[:, 0], v=edges_arr[:, 1], w=weights_arr)
        print(f"      -> Đã lưu cache cạnh Similarity tại: {cache_path}")

    del reps_norm, index, similarities, indices
    gc.collect()
    return sim_edge_dict


def load_graph_a_undirected(
    edges_path: Path,
    aux_edges_path: Path,
    id_to_idx: dict[str, int],
    use_aux_if_empty: bool = False,
) -> tuple[dict[tuple[int, int], float], str]:
    """Đọc interaction_edges.csv có hướng và gộp thành Graph A vô hướng bằng tổng trọng số 2 chiều."""
    print("[3/6] Đọc Graph A từ interaction_edges.csv (gộp trọng số cộng tổng 2 chiều)...")
    edges_df = pd.read_csv(edges_path, dtype=str)

    source_note = "interaction_edges.csv"
    if len(edges_df) == 0 and use_aux_if_empty and aux_edges_path.exists():
        print("      -> interaction_edges.csv rỗng (0 cạnh). Chuyển sang đọc file phụ trợ neighbor_edges...")
        edges_df = pd.read_csv(aux_edges_path, dtype=str)
        source_note = aux_edges_path.name

    base_edge_dict: dict[tuple[int, int], float] = {}
    if len(edges_df) > 0:
        src_col = "source" if "source" in edges_df.columns else edges_df.columns[0]
        tgt_col = "target" if "target" in edges_df.columns else edges_df.columns[1]
        w_col = "weight" if "weight" in edges_df.columns else None

        weights_iter = (
            edges_df[w_col].astype(float)
            if w_col is not None
            else np.ones(len(edges_df), dtype=float)
        )
        for src, tgt, w_val in zip(edges_df[src_col], edges_df[tgt_col], weights_iter):
            src_clean = str(src).strip().removesuffix(".0")
            tgt_clean = str(tgt).strip().removesuffix(".0")
            if src_clean in id_to_idx and tgt_clean in id_to_idx and src_clean != tgt_clean:
                if w_val > 0:
                    u_idx, v_idx = id_to_idx[src_clean], id_to_idx[tgt_clean]
                    u, v = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
                    base_edge_dict[(u, v)] = base_edge_dict.get((u, v), 0.0) + float(w_val)

    print(f"      -> Số cạnh vô hướng Graph A ({source_note}): {len(base_edge_dict):,}")
    del edges_df
    gc.collect()
    return base_edge_dict, source_note


def blend_graphs_and_export(
    account_ids: list[str],
    base_edge_dict: dict[tuple[int, int], float],
    sim_edge_dict: dict[tuple[int, int], float],
    lambda_blend: float,
    out_graphml_path: Path,
) -> tuple[dict[tuple[int, int], float], str]:
    """Bước 6: Trộn Graph A và Similarity Graph tạo Graph B và xuất GraphML qua igraph."""
    sum_base = sum(base_edge_dict.values())
    sum_sim = sum(sim_edge_dict.values())
    assert sum_sim > 0, "Tổng trọng số cạnh dương của Similarity Graph phải > 0!"

    graph_b_edges: dict[tuple[int, int], float] = {}
    if sum_base > 0:
        blend_mode = f"blended_lambda_{lambda_blend}"
        print(f"[4/6] Trộn Graph A + Similarity Graph (lambda={lambda_blend}) tạo Graph B...")
        overlap_edges = len(set(base_edge_dict.keys()) & set(sim_edge_dict.keys()))
        all_edges = set(base_edge_dict.keys()) | set(sim_edge_dict.keys())
        for e in all_edges:
            w_b = base_edge_dict.get(e, 0.0) / sum_base
            w_s = sim_edge_dict.get(e, 0.0) / sum_sim
            graph_b_edges[e] = (1.0 - lambda_blend) * w_b + lambda_blend * w_s
    else:
        blend_mode = "similarity_only_due_to_empty_graph_a"
        print(
            "[4/6] Lưu ý: Graph A có 0 cạnh tương tác nội bộ. "
            "Graph B được xây dựng từ 100% cạnh Similarity (chuẩn hóa tổng trọng số = 1.0)..."
        )
        overlap_edges = 0
        for e, w_s in sim_edge_dict.items():
            graph_b_edges[e] = w_s / sum_sim

    print(
        f"      -> Số cạnh Graph B: {len(graph_b_edges):,} "
        f"(Trùng với Graph A: {overlap_edges:,} | Bổ sung mới: {len(graph_b_edges) - len(base_edge_dict):,})"
    )

    g_export = ig.Graph(n=len(account_ids), directed=False)
    g_export.vs["id"] = account_ids
    g_export.add_edges(list(graph_b_edges.keys()))
    g_export.es["weight"] = [float(w) for w in graph_b_edges.values()]
    g_export.write_graphml(str(out_graphml_path))
    print(f"      -> Đã xuất GraphML tại: {out_graphml_path}")

    del g_export
    gc.collect()
    return graph_b_edges, blend_mode


def run_leiden_partition(
    num_nodes: int,
    edge_dict: dict[tuple[int, int], float],
    resolution: float = 1.0,
    seed: int = 42,
    iterations: int = -1,
) -> tuple[np.ndarray, dict]:
    """Bước 7-8: Chạy Leiden trên active nodes, gán community = -1 cho node cô lập."""
    if len(edge_dict) == 0:
        comm_assignments = np.full(num_nodes, -1, dtype=np.int32)
        metrics = {
            "num_nodes": int(num_nodes),
            "num_edges": 0,
            "density": 0.0,
            "active_nodes": 0,
            "isolated_nodes": int(num_nodes),
            "isolate_rate": 1.0,
            "lcc_size": 0,
            "num_communities": 0,
            "weighted_modularity": 0.0,
        }
        return comm_assignments, metrics

    active_set = set()
    for u, v in edge_dict.keys():
        active_set.add(u)
        active_set.add(v)
    active_nodes = sorted(active_set)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(active_nodes)}

    ig_edges = [(old_to_new[u], old_to_new[v]) for (u, v) in edge_dict.keys()]
    ig_weights = [float(w) for w in edge_dict.values()]

    g = ig.Graph(n=len(active_nodes), edges=ig_edges, directed=False)
    g.es["weight"] = ig_weights

    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=iterations,
    )

    comm_assignments = np.full(num_nodes, -1, dtype=np.int32)
    for new_idx, comm_id in enumerate(partition.membership):
        comm_assignments[active_nodes[new_idx]] = comm_id

    components = g.connected_components()
    lcc_size = max(components.sizes()) if len(components) > 0 else 0
    num_isolates = num_nodes - len(active_nodes)
    density = (2.0 * len(edge_dict)) / (num_nodes * (num_nodes - 1))

    metrics = {
        "num_nodes": int(num_nodes),
        "num_edges": int(len(edge_dict)),
        "density": float(density),
        "active_nodes": int(len(active_nodes)),
        "isolated_nodes": int(num_isolates),
        "isolate_rate": float(num_isolates / num_nodes),
        "lcc_size": int(lcc_size),
        "num_communities": int(len(partition)),
        "weighted_modularity": float(partition.modularity),
    }

    del g, partition, ig_edges, ig_weights
    gc.collect()
    return comm_assignments, metrics


def build_equal_edge_control(
    graph_b_edges: dict[tuple[int, int], float], account_ids: list[str], target_m: int
) -> dict[tuple[int, int], float]:
    """Đối chứng cùng số cạnh (Equal-Edge Control): Giữ top-M cạnh trọng số cao nhất, hòa điểm theo ID."""
    sorted_items = sorted(
        graph_b_edges.items(),
        key=lambda item: (-item[1], account_ids[item[0][0]], account_ids[item[0][1]]),
    )
    return dict(sorted_items[:target_m])


def analyze_posthoc_bad_distribution(
    comm_array: np.ndarray, is_bad_array: np.ndarray, graph_name: str
) -> tuple[pd.DataFrame, dict]:
    """Bước 9: Phân tích hậu nghiệm Bad Ratio, Bad Coverage và Bad Enrichment (0 = Bad, 1 = Good)."""
    total_nodes = len(is_bad_array)
    total_bad = int(is_bad_array.sum())
    baseline_bad_rate = total_bad / total_nodes

    df = pd.DataFrame({"comm": comm_array, "is_bad": is_bad_array})
    active_df = df[df["comm"] != -1]

    if len(active_df) == 0:
        empty_grouped = pd.DataFrame(
            columns=["comm", "size", "bad_count", "bad_ratio", "bad_coverage", "bad_enrichment"]
        )
        summary = {
            "graph": graph_name,
            "total_nodes_in_corpus": int(total_nodes),
            "total_bad_in_corpus": int(total_bad),
            "baseline_bad_rate": float(baseline_bad_rate),
            "bad_nodes_with_edges": 0,
            "bad_edge_coverage": 0.0,
            "max_community_size": 0,
            "median_community_size": 0.0,
            "communities_with_bad_gt_0": 0,
            "communities_enrichment_gt_1": 0,
            "top10_largest_communities": [],
        }
        return empty_grouped, summary

    bad_covered = int(active_df["is_bad"].sum())
    bad_global_coverage = bad_covered / total_bad if total_bad > 0 else 0.0

    grouped = (
        active_df.groupby("comm", as_index=False)
        .agg(size=("is_bad", "count"), bad_count=("is_bad", "sum"))
        .sort_values(by=["size", "bad_count"], ascending=[False, False])
    )

    grouped["bad_ratio"] = grouped["bad_count"] / grouped["size"]
    grouped["bad_coverage"] = grouped["bad_count"] / total_bad
    grouped["bad_enrichment"] = grouped["bad_ratio"] / baseline_bad_rate

    summary = {
        "graph": graph_name,
        "total_nodes_in_corpus": int(total_nodes),
        "total_bad_in_corpus": int(total_bad),
        "baseline_bad_rate": float(baseline_bad_rate),
        "bad_nodes_with_edges": int(bad_covered),
        "bad_edge_coverage": float(bad_global_coverage),
        "max_community_size": int(grouped["size"].max()),
        "median_community_size": float(grouped["size"].median()),
        "communities_with_bad_gt_0": int((grouped["bad_count"] > 0).sum()),
        "communities_enrichment_gt_1": int((grouped["bad_enrichment"] > 1.0).sum()),
        "top10_largest_communities": grouped.head(10).to_dict(orient="records"),
    }
    return grouped, summary


def generate_visualizations(
    base_edge_dict: dict[tuple[int, int], float],
    graph_b_edges: dict[tuple[int, int], float],
    comm_a: np.ndarray,
    comm_b: np.ndarray,
    comm_stats_a: pd.DataFrame,
    comm_stats_b: pd.DataFrame,
    is_bad: np.ndarray,
    figures_dir: Path,
) -> None:
    """Xuất 3 biểu đồ trực quan hóa cấu trúc đồ thị và phân bố cộng đồng."""
    print("[6/6] Vẽ và xuất biểu đồ trực quan hóa...")

    def _plot_subgraph(edge_dict, comm_arr, title, save_path, max_nodes=800):
        plt.figure(figsize=(10, 8), dpi=150)
        if len(edge_dict) == 0:
            plt.text(
                0.5,
                0.5,
                f"{title}\n(Đồ thị rỗng: 0 cạnh tương tác nội bộ trong dữ liệu)",
                ha="center",
                va="center",
                fontsize=12,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            return

        s = pd.Series(comm_arr)
        top_comms = s[s != -1].value_counts().head(5).index.tolist()
        selected = [i for i, c in enumerate(comm_arr) if c in top_comms][:max_nodes]
        selected_set = set(selected)
        old_to_sub = {old_i: new_i for new_i, old_i in enumerate(selected)}

        sub_edges = [
            (old_to_sub[u], old_to_sub[v])
            for (u, v) in edge_dict.keys()
            if u in selected_set and v in selected_set
        ]
        g_sub = ig.Graph(n=len(selected), edges=sub_edges, directed=False)
        layout = g_sub.layout_fruchterman_reingold()
        coords = np.array(layout.coords)

        for u_s, v_s in sub_edges:
            plt.plot(
                [coords[u_s, 0], coords[v_s, 0]],
                [coords[u_s, 1], coords[v_s, 1]],
                color="#888888",
                alpha=0.15,
                linewidth=0.5,
                zorder=1,
            )

        bad_mask = np.array([is_bad[n] == 1 for n in selected])
        plt.scatter(
            coords[~bad_mask, 0],
            coords[~bad_mask, 1],
            c="#1f77b4",
            s=12,
            alpha=0.75,
            label="Good (1)",
            zorder=2,
        )
        plt.scatter(
            coords[bad_mask, 0],
            coords[bad_mask, 1],
            c="#d62728",
            s=36,
            alpha=0.90,
            label="Bad (0)",
            zorder=3,
        )
        plt.title(f"{title}\n(Mẫu {len(selected)} nodes từ Top 5 Communities)", fontsize=11)
        plt.legend(loc="upper right")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    _plot_subgraph(
        base_edge_dict,
        comm_a,
        "Graph A (Baseline Interaction Graph)",
        figures_dir / "graph_a_visualization.png",
    )
    _plot_subgraph(
        graph_b_edges,
        comm_b,
        "Graph B (Representation-Augmented Graph)",
        figures_dir / "graph_b_visualization.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    if len(comm_stats_a) > 0:
        axes[0].hist(
            np.log10(comm_stats_a["size"]),
            bins=30,
            alpha=0.6,
            label="Graph A (Baseline)",
            color="#1f77b4",
        )
    if len(comm_stats_b) > 0:
        axes[0].hist(
            np.log10(comm_stats_b["size"]),
            bins=30,
            alpha=0.6,
            label="Graph B (Augmented)",
            color="#ff7f0e",
        )
    axes[0].set_title("Phân bố kích thước cộng đồng (Log10 Size)")
    axes[0].set_xlabel("Log10(Số tài khoản trong cộng đồng)")
    axes[0].set_ylabel("Số lượng cộng đồng")
    axes[0].legend()

    top_b_enrich = (
        comm_stats_b[comm_stats_b["size"] >= 10]
        .sort_values("bad_enrichment", ascending=False)
        .head(15)
    )
    axes[1].barh(top_b_enrich["comm"].astype(str), top_b_enrich["bad_enrichment"], color="#2ca02c")
    axes[1].invert_yaxis()
    axes[1].set_title("Top 15 Cộng đồng có Bad Enrichment cao nhất (Graph B, size >= 10)")
    axes[1].set_xlabel("Bad Enrichment = Bad Ratio / Baseline Rate")
    axes[1].set_ylabel("Community ID")

    plt.tight_layout()
    plt.savefig(figures_dir / "community_visualization.png")
    plt.close()
    print(f"      -> Đã lưu 3 biểu đồ tại: {figures_dir}")


def main() -> None:
    setup_directories(PATHS["out_dir"])

    meta_df = pd.read_csv(PATHS["e2_meta"], dtype={"account_id": str})
    nodes_df = pd.read_csv(PATHS["e3_nodes"], dtype={"account_id": str})

    assert len(meta_df) == len(nodes_df), (
        f"Số lượng tài khoản không khớp: metadata ({len(meta_df)}) vs nodes.csv ({len(nodes_df)})"
    )
    assert (meta_df["account_id"].values == nodes_df["account_id"].values).all(), (
        "Thứ tự account_id giữa honduras_account_metadata.csv và nodes.csv không trùng khớp!"
    )

    account_ids = nodes_df["account_id"].tolist()
    id_to_idx = {acc_id: idx for idx, acc_id in enumerate(account_ids)}

    labels_raw = nodes_df["label"].astype(int).values
    is_bad = (labels_raw == 0).astype(np.int32)

    # 1. Trích xuất representation 128d từ tweet_branch (hoặc dùng cache)
    out_rep_path = PATHS["out_dir"] / "representations" / "honduras_tweet_branch_128d.npy"
    reps_128d = extract_tweet_branch_representations(
        model_dir=PATHS["e1_model_dir"],
        ckpt_path=PATHS["e1_ckpt"],
        emb_path=PATHS["e2_emb"],
        out_rep_path=out_rep_path,
        batch_size=CONFIG["batch_size"],
        use_cache=CONFIG["use_cached_128d"],
    )

    # 2. Tính Cosine Similarity & lọc top-k >= threshold (hoặc dùng cache)
    sim_cache_path = (
        PATHS["out_dir"]
        / "representations"
        / f"sim_edges_top{CONFIG['top_k']}_t{int(CONFIG['sim_threshold']*100)}.npz"
    )
    sim_edge_dict = build_similarity_edges(
        reps_128d=reps_128d,
        cache_path=sim_cache_path,
        top_k=CONFIG["top_k"],
        threshold=CONFIG["sim_threshold"],
        use_cache=CONFIG["use_cached_sim_edges"],
    )
    del reps_128d
    gc.collect()

    # 3. Đọc Graph A
    base_edge_dict, graph_a_source = load_graph_a_undirected(
        edges_path=PATHS["e3_edges"],
        aux_edges_path=PATHS["e3_aux_edges"],
        id_to_idx=id_to_idx,
        use_aux_if_empty=CONFIG["use_auxiliary_neighbor_if_empty"],
    )

    # 4. Tạo Graph B & xuất GraphML
    out_graphml_path = PATHS["out_dir"] / "graphs" / "representation_augmented_graph.graphml"
    graph_b_edges, blend_mode = blend_graphs_and_export(
        account_ids=account_ids,
        base_edge_dict=base_edge_dict,
        sim_edge_dict=sim_edge_dict,
        lambda_blend=CONFIG["lambda_blend"],
        out_graphml_path=out_graphml_path,
    )
    del sim_edge_dict
    gc.collect()

    # 5. Phát hiện cộng đồng Leiden cho Graph A và Graph B
    print("[5/6] Chạy thuật toán Leiden & Phân tích hậu nghiệm cộng đồng...")
    comm_a, metrics_a = run_leiden_partition(
        num_nodes=len(account_ids),
        edge_dict=base_edge_dict,
        resolution=CONFIG["leiden_resolution"],
        seed=CONFIG["leiden_seed"],
        iterations=CONFIG["leiden_iterations"],
    )
    comm_b, metrics_b = run_leiden_partition(
        num_nodes=len(account_ids),
        edge_dict=graph_b_edges,
        resolution=CONFIG["leiden_resolution"],
        seed=CONFIG["leiden_seed"],
        iterations=CONFIG["leiden_iterations"],
    )

    # ARI trên tập active nodes của Graph A (nếu Graph A có active nodes)
    active_a_mask = comm_a != -1
    if active_a_mask.sum() > 0:
        ari_b_vs_a = float(adjusted_rand_score(comm_a[active_a_mask], comm_b[active_a_mask]))
    else:
        ari_b_vs_a = None

    comm_dir = PATHS["out_dir"] / "communities"
    pd.DataFrame({"account_id": account_ids, "community_id": comm_a}).to_csv(
        comm_dir / "graph_a_communities.csv", index=False
    )
    pd.DataFrame({"account_id": account_ids, "community_id": comm_b}).to_csv(
        comm_dir / "graph_b_communities.csv", index=False
    )

    comm_stats_a, analysis_a = analyze_posthoc_bad_distribution(comm_a, is_bad, "Graph_A_Baseline")
    comm_stats_b, analysis_b = analyze_posthoc_bad_distribution(comm_b, is_bad, "Graph_B_Augmented")

    comparison_rows = [
        {
            "Graph": "Graph A (Baseline)",
            **metrics_a,
            "bad_nodes_with_edges": analysis_a["bad_nodes_with_edges"],
            "bad_edge_coverage": analysis_a["bad_edge_coverage"],
            "communities_with_bad": analysis_a["communities_with_bad_gt_0"],
            "ari_vs_graph_a": 1.0 if active_a_mask.sum() > 0 else np.nan,
        },
        {
            "Graph": "Graph B (Augmented)",
            **metrics_b,
            "bad_nodes_with_edges": analysis_b["bad_nodes_with_edges"],
            "bad_edge_coverage": analysis_b["bad_edge_coverage"],
            "communities_with_bad": analysis_b["communities_with_bad_gt_0"],
            "ari_vs_graph_a": ari_b_vs_a if ari_b_vs_a is not None else np.nan,
        },
    ]

    graph_metrics_payload = {
        "config": CONFIG,
        "graph_a_source": graph_a_source,
        "graph_b_blend_mode": blend_mode,
        "Graph_A_Baseline": metrics_a,
        "Graph_B_Augmented": metrics_b,
        "ARI_GraphB_vs_GraphA_on_A_active": ari_b_vs_a,
    }
    community_analysis_payload = {
        "Graph_A_Baseline": analysis_a,
        "Graph_B_Augmented": analysis_b,
    }

    # Đối chứng cùng số cạnh (chỉ chạy khi Graph A có cạnh M > 0)
    if (
        CONFIG["run_equal_edge_control"]
        and len(base_edge_dict) > 0
        and len(graph_b_edges) > len(base_edge_dict)
    ):
        target_m = len(base_edge_dict)
        print(f"      -> Chạy đối chứng Equal-Edge Control (M = {target_m:,} cạnh)...")
        ctrl_edges = build_equal_edge_control(graph_b_edges, account_ids, target_m)
        comm_ctrl, metrics_ctrl = run_leiden_partition(
            num_nodes=len(account_ids),
            edge_dict=ctrl_edges,
            resolution=CONFIG["leiden_resolution"],
            seed=CONFIG["leiden_seed"],
            iterations=CONFIG["leiden_iterations"],
        )
        ari_ctrl_vs_a = float(adjusted_rand_score(comm_a[active_a_mask], comm_ctrl[active_a_mask]))
        _, analysis_ctrl = analyze_posthoc_bad_distribution(
            comm_ctrl, is_bad, "Graph_B_EqualEdgeControl"
        )

        comparison_rows.append(
            {
                "Graph": f"Graph B (Equal-Edge Control M={target_m})",
                **metrics_ctrl,
                "bad_nodes_with_edges": analysis_ctrl["bad_nodes_with_edges"],
                "bad_edge_coverage": analysis_ctrl["bad_edge_coverage"],
                "communities_with_bad": analysis_ctrl["communities_with_bad_gt_0"],
                "ari_vs_graph_a": ari_ctrl_vs_a,
            }
        )
        graph_metrics_payload["Graph_B_EqualEdgeControl"] = metrics_ctrl
        graph_metrics_payload["ARI_EqualEdgeControl_vs_GraphA_on_A_active"] = ari_ctrl_vs_a
        community_analysis_payload["Graph_B_EqualEdgeControl"] = analysis_ctrl
        del ctrl_edges, comm_ctrl
        gc.collect()

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(comm_dir / "community_comparison.csv", index=False)

    eval_dir = PATHS["out_dir"] / "evaluation"
    with open(eval_dir / "graph_metrics.json", "w", encoding="utf-8") as f:
        json.dump(graph_metrics_payload, f, indent=2, ensure_ascii=False)
    with open(eval_dir / "community_analysis.json", "w", encoding="utf-8") as f:
        json.dump(community_analysis_payload, f, indent=2, ensure_ascii=False)

    # 6. Xuất biểu đồ
    generate_visualizations(
        base_edge_dict=base_edge_dict,
        graph_b_edges=graph_b_edges,
        comm_a=comm_a,
        comm_b=comm_b,
        comm_stats_a=comm_stats_a,
        comm_stats_b=comm_stats_b,
        is_bad=is_bad,
        figures_dir=PATHS["out_dir"] / "figures",
    )

    print("\n=== HOÀN TẤT E3 ===")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()