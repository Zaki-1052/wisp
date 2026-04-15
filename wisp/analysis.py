# wisp/analysis.py
import os
from collections import Counter

import networkx as nx
import numpy as np
from loguru import logger


def calculate_all_shortest_paths(G: nx.Graph) -> dict[tuple[int, int], tuple[float, list[int]]]:
    """Compute shortest paths between all node pairs using Dijkstra's algorithm.

    Args:
        G: nx.Graph with weighted edges built from the correlation matrix.

    Returns:
        Dictionary mapping (source, sink) tuples to (length, path) tuples.
    """
    logger.info("Computing all-pairs shortest paths...")
    all_paths = {}
    length_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    path_paths = dict(nx.all_pairs_dijkstra_path(G, weight="weight"))

    for source, lengths in length_paths.items():
        for sink, length in lengths.items():
            if source != sink:
                path = path_paths[source][sink]
                all_paths[(source, sink)] = (length, path)

    logger.info(f"Computed {len(all_paths)} shortest paths.")
    return all_paths


def analyze_shortest_paths(
    all_paths: dict[tuple[int, int], tuple[float, list[int]]],
    output_dir: str,
    path_usage_threshold: float = 0.1,
    centrality_threshold: float = 0.1,
    edge_criticality_threshold: float = 0.1,
) -> dict:
    """Analyze all-pairs shortest path data and write results to files.

    Args:
        all_paths: Dictionary from calculate_all_shortest_paths.
        output_dir: Directory to write output files into.
        path_usage_threshold: Fraction of paths an edge must appear in for
            usage reporting.
        centrality_threshold: Fraction of paths a node must appear in to be
            considered a hub.
        edge_criticality_threshold: Fraction of paths an edge must appear in
            to be considered critical.

    Returns:
        Dictionary with analysis results: average_path_length, hub_nodes,
        critical_edges, path_length_distribution, detailed_node_usage,
        detailed_edge_usage.
    """
    edge_usage: Counter = Counter()
    node_usage: Counter = Counter()
    path_lengths: list[float] = []
    total_paths = len(all_paths)

    for (src, sink), (length, path) in all_paths.items():
        if path:
            path_lengths.append(length)
            node_usage.update(path)
            for start, end in zip(path[:-1], path[1:]):
                edge_usage[(start, end)] += 1

    critical_edges = {
        k: v
        for k, v in edge_usage.items()
        if v / total_paths > edge_criticality_threshold
    }
    hub_nodes = {
        k: v for k, v in node_usage.items() if v / total_paths > centrality_threshold
    }

    avg_path_length = np.mean(path_lengths) if path_lengths else float("inf")
    path_length_distribution = np.histogram(path_lengths, bins=10)

    _write_path_lengths(output_dir, avg_path_length, path_length_distribution)
    _write_node_usage(output_dir, node_usage)
    _write_edge_usage(output_dir, edge_usage)
    _write_critical_edges(output_dir, critical_edges)
    _write_hub_nodes(output_dir, hub_nodes)

    logger.info(
        f"Analysis complete: {len(hub_nodes)} hub nodes, "
        f"{len(critical_edges)} critical edges, "
        f"avg path length {avg_path_length:.4f}"
    )

    return {
        "average_path_length": avg_path_length,
        "hub_nodes": list(hub_nodes.keys()),
        "critical_edges": list(critical_edges.keys()),
        "path_length_distribution": path_length_distribution,
        "detailed_node_usage": node_usage,
        "detailed_edge_usage": edge_usage,
    }


def _write_path_lengths(output_dir, avg_path_length, distribution):
    with open(os.path.join(output_dir, "path_lengths.txt"), "w", encoding="utf-8") as f:
        f.write(f"Average Path Length: {avg_path_length}\n")
        counts, bin_edges = distribution
        f.write("Path Length Distribution (histogram):\n")
        for i, count in enumerate(counts):
            f.write(f"  {bin_edges[i]:.4f} - {bin_edges[i+1]:.4f}: {count}\n")


def _write_node_usage(output_dir, node_usage):
    with open(os.path.join(output_dir, "node_usage.txt"), "w", encoding="utf-8") as f:
        for node, usage in sorted(node_usage.items()):
            f.write(f"Node {node}: {usage}\n")


def _write_edge_usage(output_dir, edge_usage):
    with open(os.path.join(output_dir, "edge_usage.txt"), "w", encoding="utf-8") as f:
        for (start, end), usage in sorted(edge_usage.items()):
            f.write(f"Edge {start} -> {end}: {usage}\n")


def _write_critical_edges(output_dir, critical_edges):
    with open(
        os.path.join(output_dir, "critical_edges.txt"), "w", encoding="utf-8"
    ) as f:
        for (start, end), usage in sorted(
            critical_edges.items(), key=lambda x: x[1], reverse=True
        ):
            f.write(f"Critical Edge {start} -> {end} (usage: {usage})\n")


def _write_hub_nodes(output_dir, hub_nodes):
    with open(os.path.join(output_dir, "hub_nodes.txt"), "w", encoding="utf-8") as f:
        for node, usage in sorted(
            hub_nodes.items(), key=lambda x: x[1], reverse=True
        ):
            f.write(f"Hub Node: {node} (usage: {usage})\n")
