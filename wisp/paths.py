from typing import Any

import copy
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Collection, MutableMapping
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import numpy.typing as npt
from loguru import logger


def get_log_n_paths(graph, cutoff_length):
    # Calculate the average branching factor
    total_edges = graph.number_of_edges()
    total_nodes = graph.number_of_nodes()
    avg_branching_factor = total_edges / total_nodes if total_nodes else 0

    # Use logarithms to avoid overflow
    # Check if avg_branching_factor is greater than 1 to avoid log(0) or negative values
    if avg_branching_factor > 1:
        log_estimated_paths = cutoff_length * np.log(avg_branching_factor)
    else:
        # If the avg_branching_factor is 1 or less, the growth is linear or
        # non-existent, not exponential
        log_estimated_paths = 0

    logger.debug(f"log(estimated_n_paths) = {log_estimated_paths}")

    return log_estimated_paths


def expand_growing_paths_one_step(
    paths_growing_out_from_source,
    full_paths_from_start_to_sink,
    cutoff,
    sink,
    G,
):
    """Expand the first path growing out from the source by one step, to the
       neighbors of its terminal node.

    The expanded path is removed from `paths_growing_out_from_source` and replaced
    by one longer path per eligible neighbor, so repeated calls walk the whole tree.

    Args:
        paths_growing_out_from_source: a list of paths, where each path is
            represented by a list. The first item in each path is the length of
            the path (float). The remaining items are the indices of the nodes
            in the path (int). Must not be empty.
        full_paths_from_start_to_sink: a growing list of identified paths that
            connect the source and the sink, where each path is formatted as above.
        cutoff: a numpy array containing a single element (float), the length
            cutoff. Paths with lengths greater than the cutoff will be ignored.
        sink: the index of the sink (int)
        G: a nx.Graph object describing the connectivity of the different nodes
    """

    path = paths_growing_out_from_source.pop(0)

    if path[0] > cutoff:
        # Because if the path is already greater than the cutoff, no use
        # continuing to branch out, since subsequent branches will be longer.
        return

    if path[-1] == sink:
        full_paths_from_start_to_sink.append(path)
        return

    # Sink not yet reached, but paths still short enough. So add new paths, same
    # as old, but with neighboring element appended.
    for i, node_neighbor in enumerate(G.neighbors(path[-1])):
        if node_neighbor not in path:
            expanded_path = path[:]
            expanded_path.append(node_neighbor)
            expanded_path[0] = (
                expanded_path[0]
                + G.edges[expanded_path[-2], expanded_path[-1]]["weight"]
            )
            paths_growing_out_from_source.insert(i, expanded_path)


# Populated in each worker process by _init_path_search_worker. Holding the search
# parameters in a global means the graph is pickled once per worker rather than
# once per branch.
_PATH_SEARCH_CONTEXT: tuple | None = None


def _init_path_search_worker(cutoff, sink, G) -> None:
    """Store the search parameters in this worker process."""
    global _PATH_SEARCH_CONTEXT  # pylint: disable=global-statement
    _PATH_SEARCH_CONTEXT = (cutoff, sink, G)


def _find_paths_from_branch(branch: list) -> list:
    """Expand a single partial path until every full source-to-sink path is found.

    Args:
        branch: a list corresponding to a partial path. The first item is the
            length of the path (float). The remaining items are the indices of
            the nodes in the path (int).

    Returns:
        A list of the full source-to-sink paths that grow out of this branch.
    """
    if _PATH_SEARCH_CONTEXT is None:
        raise RuntimeError("Path-finding worker process was never given a graph.")
    cutoff, sink, G = _PATH_SEARCH_CONTEXT

    paths_growing_out_from_source = [branch]
    full_paths_from_start_to_sink: list = []

    while paths_growing_out_from_source:
        expand_growing_paths_one_step(
            paths_growing_out_from_source,
            full_paths_from_start_to_sink,
            cutoff,
            sink,
            G,
        )

    return full_paths_from_start_to_sink


def find_paths_in_parallel(
    branches: Collection, cutoff, sink, G, n_cores: int | None = None
) -> list:
    """Expand partial paths into full source-to-sink paths using multiple processes.

    Each branch is expanded independently, so they are handed out to worker
    processes one at a time.

    Args:
        branches: the partial paths to expand, each a list whose first item is the
            path length (float) and whose remaining items are node indices (int).
        cutoff: a numpy array containing a single element (float), the length
            cutoff. Paths with lengths greater than the cutoff will be ignored.
        sink: the index of the sink (int)
        G: a nx.Graph object describing the connectivity of the different nodes
        n_cores: the number of processes to use. Defaults to every available core.

    Returns:
        A flat list of every full source-to-sink path found, where each path is
        represented by a list. The first item is the length of the path (float).
        The remaining items are the indices of the nodes in the path (int).
    """
    if n_cores is None:
        n_cores = mp.cpu_count()
    n_workers = max(1, min(n_cores, len(branches)))
    logger.debug(f"Expanding {len(branches)} branches on {n_workers} cores")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_path_search_worker,
        initargs=(cutoff, sink, G),
    ) as pool:
        branch_results = pool.map(_find_paths_from_branch, branches)
        return [path for branch_paths in branch_results for path in branch_paths]


class GetPaths:
    """Get the paths from a list of sources to a list of sinks"""

    def __init__(
        self,
        corr_matrix: npt.NDArray[np.floating],
        srcs: Collection[int],
        snks: Collection[int],
        context: MutableMapping[str, Any],
        residue_keys: npt.ArrayLike,
    ):
        """Identify paths that link the source and the sink and order them by their
        lengths.

        Args:
            corr_matrix: a np.array, the calculated correlation matrix
            srcs: a list of ints, the indices of the sources for path finding
            snks: a list of ints, the indices of the sinks for path finding
            context: the WISP context dictionary
            residue_keys: a list containing string representations of each residue
        """
        # populate graph nodes and weighted edges
        G = nx.Graph(incoming_graph_data=corr_matrix)

        # first calculate length of shortest path between any source and sink
        logger.info("Calculating paths...")
        logger.info(
            "Calculating the shortest path between any of the specified sources and any of the specified sinks...",
        )
        shortest_length, shortest_path = self.get_shortest_path_length(
            corr_matrix, srcs, snks, G
        )
        logger.info(
            f"The shortest path has length {str(shortest_length)}",
        )

        path = [shortest_length]
        path.extend(shortest_path)
        pths = [
            path
        ]  # need to create this initial path in case only one path is requrested

        cutoff = shortest_length

        # Check for comb explosion
        log_n_paths = get_log_n_paths(G, cutoff)
        if log_n_paths > np.log(context["n_paths_max"]):
            logger.error(
                f"Estimated number of paths is greater than {context['n_paths_max']}"
            )
            logger.error("Please increase n_paths_max to proceed.")
            logger.error("Terminating calculation.")
            sys.exit(1)

        cutoff_yields_max_num_paths_below_target = 0
        cutoff_yields_min_num_paths_above_target = 1000000.0

        # first step, keep incrementing a little until you have more than the desired number of paths
        logger.info(
            "Identifying the cutoff required to produce "
            + str(context["n_paths"])
            + " paths...",
        )
        num_paths = 1
        while num_paths < context["n_paths"]:
            logger.info(f"Testing the cutoff {str(cutoff)}...")
            cutoff_in_array = np.array([cutoff], np.float64)
            pths = self.remove_redundant_paths(
                self.get_paths_between_multiple_endpoints(
                    cutoff_in_array, corr_matrix, srcs, snks, G, context
                )
            )
            num_paths = len(pths)

            logger.info(
                f"The cutoff {str(cutoff)} produces {num_paths} paths...",
            )

            if (
                num_paths < context["n_paths"]
                and cutoff > cutoff_yields_max_num_paths_below_target
            ):
                cutoff_yields_max_num_paths_below_target = cutoff
            if (
                num_paths > context["n_paths"]
                and cutoff < cutoff_yields_min_num_paths_above_target
            ):
                cutoff_yields_min_num_paths_above_target = cutoff

            # Original code adds .1 each time... but this may be to fast for
            # some systems and very slow for others... lets try increasing by
            # a percentage of the minimum path length instead... ideally this
            # could be an input parameter in the future.
            cutoff = cutoff + shortest_length * 0.1

        pths = self.remove_redundant_paths(pths)

        pths.sort()  # sort the paths by length

        if num_paths != context["n_paths"]:  # so further refinement is needed
            pths = pths[: context["n_paths"]]
            logger.info(
                "Keeping the first " + str(context["n_paths"]) + " of these paths...",
            )

        self.paths_description = ""

        self.paths_description = (
            self.paths_description + "\n# Output identified paths" + "\n"
        )
        index = 1

        for path in pths:
            self.paths_description = (
                f"{self.paths_description}Path {str(index)}:" + "\n"
            )
            self.paths_description = (
                f"{self.paths_description}   Length: {str(path[0])}" + "\n"
            )
            self.paths_description = (
                f"{self.paths_description}   Nodes: "
                + " - ".join([residue_keys[item] for item in path[1:]])
                + "\n"
            )
            index = index + 1

        if context["write_formatted_paths"]:
            formatted_paths_path = os.path.join(
                context["output_dir"], "simply_formatted_paths.txt"
            )
            with open(formatted_paths_path, "w", encoding="utf-8") as f:
                f.writelines(" ".join([str(item) for item in path]) + "\n")

        self.paths = pths

    def remove_redundant_paths(self, pths):
        """Removes redundant paths

        Args:
            pths: a list of paths

        Returns:
            A list of paths with the redundant ones eliminated.
        """

        if len(pths) == 1:
            # no reason to check if there's only one
            return pths

        for indx1 in range(len(pths) - 1):
            path1 = pths[indx1]
            if path1 is not None:
                for indx2 in range(indx1 + 1, len(pths)):
                    path2 = pths[indx2]
                    if path2 is not None and len(path1) == len(
                        path2
                    ):  # paths are the same length
                        pth1 = copy.deepcopy(path1[1:])
                        pth2 = copy.deepcopy(path2[1:])

                        if pth1[0] < pth1[-1]:
                            pth1.reverse()
                        if pth2[0] < pth2[-1]:
                            pth2.reverse()

                        if pth1 == pth2:
                            pths[indx2] = None

        while None in pths:
            pths.remove(None)

        return pths

    def get_shortest_path_length(
        self, corr_matrix, srcs, snks, G
    ):  # where sources and sinks are lists
        """Identify the length of the shortest path connecting any of the sources and any of the sinks

        Args:
            corr_matrix: a np.array, the calculated correlation matrix
            srcs: a list of ints, the indices of the sources for path finding
            snks: a list of ints, the indices of the sinks for path finding
            G: a nx.Graph object describing the connectivity of the different nodes

        Returns:
            a float, the length of the shortest path, and a list of ints corresponding
            to the nodes of the shortest path.
        """

        shortest_length = 99999999.999
        shortest_path = []

        for source in srcs:
            for sink in snks:
                if source != sink:  # important to avoid this situation
                    short_path = nx.dijkstra_path(G, source, sink, weight="weight")
                    length = self.get_length_of_path(short_path, corr_matrix)
                    if length < shortest_length:
                        shortest_length = length
                        shortest_path = short_path
        return shortest_length, shortest_path

    def get_length_of_path(self, path, corr_matrix):
        """Calculate the length of a path

        Args:
            path: a list of ints, the indices of the path
            corr_matrix: a np.array, the calculated correlation matrix

        Returns:
            a float, the length of the path
        """

        length = 0.0
        for t in range(len(path) - 1):
            length = length + corr_matrix[path[t], path[t + 1]]
        return length

    def get_paths_between_multiple_endpoints(
        self, cutoff, corr_matrix, srcs, snks, G, context
    ):  # where sources and sinks are lists
        """Get paths between sinks and sources

        Args:
            cutoff: a np.array containing a single float, the cutoffspecifying the maximum permissible path length
            corr_matrix: a np.array, the calculated correlation matrix
            srcs: a list of ints, the indices of the sources for path finding
            snks: a list of ints, the indices of the sinks for path finding
            G: a nx.Graph object describing the connectivity of the different nodes
            context: the WISP context dictionary

        Returns:
            a list of paths, where each path is represented by a list. The first item in each path is the length
            of the path (float). The remaining items are the indices of the nodes in the path (int).
        """

        pths = []
        for source in srcs:
            for sink in snks:
                if source != sink:  # avoid this situation
                    pths.extend(
                        self.get_paths_fixed_endpoints(
                            cutoff, corr_matrix, source, sink, G, context
                        )
                    )
        return pths

    def get_paths_fixed_endpoints(self, cutoff, corr_matrix, source, sink, G, context):
        """Get paths between a single sink and a single source

        Args:
            cutoff: a np.array containing a single float, the cutoff specifying the
                maximum permissible path length
            corr_matrix: a np.array, the calculated correlation matrix
            source: the index of the source for path finding
            sink: the index of the sink for path finding
            G: a nx.Graph object describing the connectivity of the different nodes
            context: the WISP context dictionary

        Returns:
            a list of paths, where each path is represented by a list. The first item
            in each path is the length of the path (float). The remaining items are
            the indices of the nodes in the path (int).
        """

        if source == sink:
            return []

        source_lengths, source_paths = nx.single_source_dijkstra(
            G, source, target=None, cutoff=None, weight="weight"
        )
        sink_lengths, sink_paths = nx.single_source_dijkstra(
            G, sink, target=None, cutoff=None, weight="weight"
        )

        so_l = [source_lengths[key] for key in source_lengths.keys()]
        so_p = [source_paths[key] for key in source_paths.keys()]
        si_l = [sink_lengths[key] for key in sink_lengths.keys()]
        si_p = [sink_paths[key] for key in sink_paths.keys()]

        check_list_1 = []
        check_list_2 = []
        for i in range(len(so_l)):
            check_list_1.extend([so_p[i][-1]])
            check_list_2.extend([si_p[i][-1]])

        node_list = []
        dijkstra_list = []
        upper_minimum_length = 0
        if not set(check_list_1).difference(check_list_2):
            for i, _ in enumerate(so_l):
                if so_l[i] + si_l[i] <= cutoff:
                    node_list.extend(so_p[i][:])
                    node_list.extend(si_p[i][:])
                    si_pReversed = si_p[i][:]
                    si_pReversed.reverse()
                    temp_path = so_p[i][:] + si_pReversed[1:]
                    temp_length = so_l[i] + si_l[i]
                    dijkstra_list.append(temp_path)
                    if (so_l[i] + si_l[i]) > upper_minimum_length:
                        upper_minimum_length = temp_length
        else:
            logger.critical("paths do not match up")

        unique_nodes = list(set(node_list))
        unique_nodes.sort()

        node_length = len(unique_nodes)
        new_matrix = np.zeros((len(corr_matrix), len(corr_matrix)))

        for i in range(node_length):
            for j in range(node_length):
                new_matrix[unique_nodes[i]][unique_nodes[j]] = corr_matrix[
                    unique_nodes[i]
                ][unique_nodes[j]]

        corr_matrix = new_matrix
        G = nx.Graph(incoming_graph_data=corr_matrix, labels=unique_nodes)

        length = 0.0
        paths_growing_out_from_source = [[length, source]]
        full_paths_from_start_to_sink = []

        # This is essentially this list-addition replacement for a recursive
        # algorithm you've envisioned.
        # To parallelize, just get the first N branches, and send them off to each node.
        # Rest of branches filled out in separate processes.

        if context["n_cores"] == 1:
            while paths_growing_out_from_source:
                expand_growing_paths_one_step(
                    paths_growing_out_from_source,
                    full_paths_from_start_to_sink,
                    cutoff,
                    sink,
                    G,
                )
        else:
            # just get some of the initial paths on a single processor
            logger.info(
                "Starting serial portion of path-finding algorithm (will run for "
                + str(context["seconds_to_wait_before_parallelizing_path_finding"])
                + " seconds)...",
            )
            atime = time.time()
            while (
                paths_growing_out_from_source
                and time.time() - atime
                < context["seconds_to_wait_before_parallelizing_path_finding"]
            ):
                expand_growing_paths_one_step(
                    paths_growing_out_from_source,
                    full_paths_from_start_to_sink,
                    cutoff,
                    sink,
                    G,
                )

            # ok, so having generated just a first few, divy up those among multiple processors
            if paths_growing_out_from_source:  # in case you've already finished
                logger.info(
                    "Starting parallel portion of path-finding algorithm running on "
                    + str(context["n_cores"])
                    + " processors...",
                )
                full_paths_from_start_to_sink.extend(
                    find_paths_in_parallel(
                        paths_growing_out_from_source,
                        cutoff,
                        sink,
                        G,
                        context["n_cores"],
                    )
                )
            else:
                logger.info(
                    "(All paths found during serial path finding; parallelization not required)",
                )

        full_paths_from_start_to_sink.sort()

        pths = []

        for full_path_from_start_to_sink in full_paths_from_start_to_sink:
            pths.append(full_path_from_start_to_sink)

        return pths
