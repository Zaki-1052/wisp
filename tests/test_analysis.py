# tests/test_analysis.py
import re

import networkx as nx
import pytest

from wisp.analysis import analyze_shortest_paths, calculate_all_shortest_paths


def make_test_graph() -> nx.Graph:
    """Build a small weighted graph whose shortest paths are all unambiguous.

    Every shortest path in this graph is strictly shorter than its alternatives,
    so results do not depend on how ties are broken.
    """
    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (1, 3, 2.5),
            (0, 3, 5.0),
        ]
    )
    return G


def test_calculate_all_shortest_paths_serial():
    G = make_test_graph()

    all_paths = calculate_all_shortest_paths(G, n_cores=1)

    n_nodes = G.number_of_nodes()
    assert len(all_paths) == n_nodes * (n_nodes - 1)

    # 0-1-2-3-4 (4.0) beats 0-1-3-4 (4.5) and 0-3-4 (6.0).
    length, path = all_paths[(0, 4)]
    assert path == [0, 1, 2, 3, 4]
    assert length == pytest.approx(4.0)

    # 0-1-2-3 (3.0) beats 0-1-3 (3.5) and 0-3 (5.0).
    length, path = all_paths[(0, 3)]
    assert path == [0, 1, 2, 3]
    assert length == pytest.approx(3.0)


def test_calculate_all_shortest_paths_parallel_matches_serial():
    G = make_test_graph()

    serial = calculate_all_shortest_paths(G, n_cores=1)
    parallel = calculate_all_shortest_paths(G, n_cores=2)

    assert serial.keys() == parallel.keys()
    for key, (length, path) in serial.items():
        parallel_length, parallel_path = parallel[key]
        assert parallel_length == pytest.approx(length)
        assert parallel_path == path


def test_analyze_shortest_paths_writes_output_files(tmp_path):
    all_paths = calculate_all_shortest_paths(make_test_graph())

    analyze_shortest_paths(all_paths, str(tmp_path))

    for filename in (
        "path_lengths.txt",
        "node_usage.txt",
        "edge_usage.txt",
        "critical_edges.txt",
        "hub_nodes.txt",
    ):
        assert (tmp_path / filename).is_file(), f"{filename} was not written"

    # The usage counts are what downstream parsing keys off of, so they must be
    # present on every line.
    hub_lines = (tmp_path / "hub_nodes.txt").read_text().splitlines()
    assert hub_lines, "expected at least one hub node"
    for line in hub_lines:
        assert re.fullmatch(r"Hub Node: \d+ \(usage: \d+\)", line), line

    edge_lines = (tmp_path / "critical_edges.txt").read_text().splitlines()
    assert edge_lines, "expected at least one critical edge"
    for line in edge_lines:
        assert re.fullmatch(r"Critical Edge \d+ -> \d+ \(usage: \d+\)", line), line


def test_analyze_shortest_paths_preserves_usage_counts(tmp_path):
    all_paths = calculate_all_shortest_paths(make_test_graph())

    results = analyze_shortest_paths(all_paths, str(tmp_path))

    assert isinstance(results["hub_nodes"], dict)
    assert isinstance(results["critical_edges"], dict)
    assert results["hub_nodes"], "expected at least one hub node"
    assert results["critical_edges"], "expected at least one critical edge"

    # Each reported count must match the tally it was filtered from.
    for node, count in results["hub_nodes"].items():
        assert results["detailed_node_usage"][node] == count
    for edge, count in results["critical_edges"].items():
        assert results["detailed_edge_usage"][edge] == count


def test_analyze_shortest_paths_thresholds_filter(tmp_path):
    all_paths = calculate_all_shortest_paths(make_test_graph())

    permissive = analyze_shortest_paths(
        all_paths, str(tmp_path), centrality_threshold=0.0
    )
    strict = analyze_shortest_paths(all_paths, str(tmp_path), centrality_threshold=0.99)

    assert len(permissive["hub_nodes"]) == make_test_graph().number_of_nodes()
    # No node appears in every single path, so nothing clears a 99% threshold.
    assert strict["hub_nodes"] == {}
