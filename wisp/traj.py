import multiprocessing as mp
from collections.abc import Collection
from concurrent.futures import ProcessPoolExecutor

from loguru import logger

from .structure import Molecule


# --- first, to generate a covariance matrix ---#
class FrameDataCollector:
    """Accumulate node locations and summed coordinates across PDB trajectory frames."""

    def __init__(self):
        self.summed_coordinates = None
        self.nodes: dict = {}

    def add_frame(self, context_and_pdb_lines) -> None:
        """Process a single PDB frame: identify the relevant nodes.

        Args:
            context_and_pdb_lines: a tuple whose first item is the WISP context
                dictionary and whose second item is a list of strings representing
                the PDB frame to be processed, where each string contains a PDB
                ATOM or HETATM entry.
        """
        context, pdb_lines = context_and_pdb_lines

        # now load the frame into its own Molecule object
        pdb = Molecule()
        pdb.load_pdb_from_list(pdb_lines)

        if self.summed_coordinates is None:
            self.summed_coordinates = pdb.coordinates
        else:
            self.summed_coordinates = self.summed_coordinates + pdb.coordinates

        pdb.map_atoms_to_residues()
        pdb.map_nodes_to_residues(context["node_definition"])

        for index, residue_iden in enumerate(pdb.residue_identifiers_in_order):
            if residue_iden in self.nodes:
                self.nodes[residue_iden].append(pdb.nodes[index])
            else:
                self.nodes[residue_iden] = [pdb.nodes[index]]

    def merge(self, other: "FrameDataCollector") -> None:
        """Fold the frames accumulated by another collector into this one.

        Args:
            other: a collector whose frames should be added to this one. Merging
                an empty collector is a no-op.
        """
        if other.summed_coordinates is None:
            return

        if self.summed_coordinates is None:
            self.summed_coordinates = other.summed_coordinates
        else:
            self.summed_coordinates = self.summed_coordinates + other.summed_coordinates

        for residue_iden, node_list in other.nodes.items():
            if residue_iden in self.nodes:
                self.nodes[residue_iden].extend(node_list)
            else:
                self.nodes[residue_iden] = node_list


def _collect_data_from_frame_chunk(frames: Collection) -> FrameDataCollector:
    """Accumulate one chunk of frames inside a worker process."""
    collector = FrameDataCollector()
    for frame in frames:
        collector.add_frame(frame)
    return collector


def collect_data_from_frames_in_parallel(
    inputs: Collection, n_cores: int | None = None
) -> FrameDataCollector:
    """Collect node data from PDB frames using multiple processes.

    Each worker accumulates its own chunk of frames so that only one result per
    worker travels back to this process.

    Args:
        inputs: the frames to process, as a list of (context, pdb_lines) tuples.
        n_cores: the number of processes to use. Defaults to every available core.

    Returns:
        A collector holding the summed coordinates and node lists for every frame
        in `inputs`.
    """
    if n_cores is None:
        n_cores = mp.cpu_count()
    n_workers = max(1, min(n_cores, len(inputs)))
    logger.debug(f"Collecting data from {len(inputs)} frames on {n_workers} cores")

    # Deal the frames round-robin so each worker gets a comparable share.
    frames = list(inputs)
    chunks = [frames[i::n_workers] for i in range(n_workers)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        chunk_collectors = pool.map(_collect_data_from_frame_chunk, chunks)

        combined = FrameDataCollector()
        for chunk_collector in chunk_collectors:
            combined.merge(chunk_collector)

    return combined
