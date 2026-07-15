import os
import time
import argparse
import numpy as np
import pickle
from wisp.run import run_wisp
from wisp.contexts import ContextManager
from wisp.structure import Atom, Molecule
from wisp.traj import multi_threading_to_collect_data_from_frames, collect_data_from_frames
from wisp.paths import multi_threading_find_paths, GetPaths
from wisp.utils import GetCovarianceMatrix
from wisp.viz import Visualize
from wisp.io import output_dir_info, UserInput, log

### SET DIRECTORIES ###
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(WORKING_DIR, "input/")
WRITING_DIR = os.path.join(WORKING_DIR,"tmp/")

if __name__ == "__main__":
    program_start_time = time.time()

    # get the commandline parameters
    argparser = argparse.ArgumentParser(description="WISP-SPM:Shortest Path Method using the Weighted Implementation of Suboptimal Paths")
    argparser.add_argument(
        "-prmtop",
        type=str,
        required=False,
        help="Topology file in AMBER format (prmtop)",)
    
    argparser.add_argument(
        "-nc",
        type=str,
        required=False,
        help="Trajectory file in AMBER format (nc)",
    )
    argparser.add_argument(
        "-pdb",
        type=str,
        required=True,
        help="The filename of the multi-frame PDB to analyze. Individual frames should be separated by 'END' or 'ENDMDL' lines.",
    )
    argparser.add_argument(
        "-distmat",
        type=str,
        required=True,
        help="The filename of the pre-computed distance matrix to use.",
    )
    argparser.add_argument(
        "-corrmat",
        type=str,
        required=True,
        help="The filename of the pre-computed correlation matrix to use.",
    )
    args = argparser.parse_args()
    parameters = UserInput()
    parameters.parameters["pdb_trajectory_filename"] = args.pdb
    parameters.parameters["user_specified_functionalized_matrix_filename"] = args.corrmat
    parameters.parameters["user_specified_contact_map_filename"] = args.distmat

    # compute the correlation matrix
    if parameters["load_wisp_saved_matrix"] == "TRUE":
        correlation_matrix_object = pickle.load(
            open(parameters["wisp_saved_matrix_filename"], "rb")
        )  # load the matrix instead of generating
    else:
        correlation_matrix_object = GetCovarianceMatrix(
            parameters
        )  # so generate the matrix instead of loading it
    correlation_matrix = correlation_matrix_object.correlations

    # always save a copy of the correlation matrix, regardless of how it was loaded/generated
    pickle.dump(
        correlation_matrix_object,
        open(
            parameters["output_directory"]
            + "functionalized_matrix_with_contact_map_applied.pickle",
            "wb",
        ),
    )

    # now get the source and sink locations from the parameters
    sources = correlation_matrix_object.convert_list_of_residue_keys_to_residue_indices(
        parameters["source_residues"]
    )
    sinks = correlation_matrix_object.convert_list_of_residue_keys_to_residue_indices(
        parameters["sink_residues"]
    )

    # compute the paths
    paths = GetPaths(
        correlation_matrix,
        sources,
        sinks,
        parameters,
        correlation_matrix_object.average_pdb.residue_identifiers_in_order,
    )

    # create the visualization
    #vis = Visualize(parameters, correlation_matrix_object, paths)

    # provide the user information about the generated files
    output_dir_info(parameters)

    log(
        "\n# Program execution time: "
        + str(time.time() - program_start_time)
        + " seconds",
        parameters["logfile"],
    )

    parameters["logfile"].close()