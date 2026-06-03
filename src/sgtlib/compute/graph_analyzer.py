# SPDX-License-Identifier: GNU GPL v3
"""
Compute graph theory metrics
"""
import os
import math
import time
import copy
import datetime
import logging
import gudhi
import multiprocessing
import numpy as np
import scipy as sp
import pandas as pd
import igraph as ig
import networkx as nx
import matplotlib.table as tbl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from collections import defaultdict
from statistics import stdev, StatisticsError
from matplotlib.backends.backend_pdf import PdfPages
from networkx import exception

from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality, eigenvector_centrality
from networkx.algorithms import average_node_connectivity, global_efficiency, clustering
from networkx.algorithms import degree_assortativity_coefficient
from networkx.algorithms.flow import maximum_flow
from networkx.algorithms.distance_measures import diameter, periphery
from networkx.algorithms.wiener import wiener_index

from ..networks.fiber_network import FiberNetworkBuilder
from ..imaging.image_processor import ImageProcessor
from ..utils.config_loader import load_gtc_configs
from ..utils.gen_plots import CurveFitModels
from ..utils.sgt_utils import get_num_cores, AbortException, ProgressUpdate, ProgressData

logger = logging.getLogger("SGT App")

# WE ARE USING CPU BECAUSE CuPy generates some errors - yet to be resolved.
COMPUTING_DEVICE = "CPU"
"""
try:
    import sys
    import cupy as cp

    # Check for GPU
    test = cp.cuda.Device(0).compute_capability
    # Check for CUDA_PATH in environment variables
    cuda_path = os.getenv("CUDA_PATH")
    print(cuda_path)
    if cuda_path:
        xp = np  # Use CuPy for GPU
        COMPUTING_DEVICE = "GPU"
        logging.info("Using GPU with CuPy!", extra={'user': 'SGT Logs'})
    else:
        logging.info(
            "Please add CUDA_PATH to System environment variables OR install 'NVIDIA GPU Computing Toolkit'\nvia: https://developer.nvidia.com/cuda-downloads",
            extra={'user': 'SGT Logs'})
        raise ImportError("Please add CUDA_PATH to System environment variables.")
except (ImportError, NameError, AttributeError):
    xp = np  # Fallback to NumPy for CPU
    logging.info("Using CPU with NumPy!", extra={'user': 'SGT Logs'})
except cp.cuda.runtime.CUDARuntimeError:
    xp = np  # Fallback to NumPy for CPU
    logging.info("Using CPU with NumPy!", extra={'user': 'SGT Logs'})
"""


def _worker_vertex_connectivity(ig_graph_obj, i, j):
    try:
        lnc = ig_graph_obj.vertex_connectivity(source=i, target=j, neighbors="negative")
        return lnc if lnc != -1 else None
    except Exception as err:
        logging.exception("Computing iGraph ANC Error: %s", err, extra={'user': 'SGT Logs'})
        return None


class GraphAnalyzer(ProgressUpdate):
    """
    A class that computes all the user-selected graph theory metrics and writes the results in a PDF file.

    Args:
        :param imp: Image Processor object.
        allow_multiprocessing: a decision to allow multiprocessing computing.
    """

    def __init__(self, imp: ImageProcessor, allow_multiprocessing: bool = False, use_igraph: bool = True):
        """
        A class that computes all the user-selected graph theory metrics and writes the results in a PDF file.

        :param imp: Image Processor object.
        :param allow_multiprocessing: Allows multiprocessing computing.
        :param use_igraph: Whether to use igraph C library module.

        >>> i_path = "path/to/image"
        >>> cfg_file = "path/to/sgt_configs.ini"
        >>>
        >>> def print_update(progress_val, progress_msg):
        ...     print(f"{progress_val}: {progress_msg}")
        >>>
        >>> ntwk_obj, _ = ImageProcessor.from_image_file(i_path)
        >>> metrics_obj = GraphAnalyzer(ntwk_obj)
        >>> GraphAnalyzer.safe_run_analyzer(metrics_obj, print_update, save_to_pdf=True)

        """
        super(GraphAnalyzer, self).__init__()
        self._configs: dict = load_gtc_configs(imp.config_file)  # graph theory computation parameters and options.
        self._props: list = []
        self._allow_mp: bool = allow_multiprocessing
        self._use_igraph: bool = use_igraph
        self._ntwk_p: ImageProcessor = imp
        self._plot_figures: list | None = None
        self._results_df:   None | pd.DataFrame = None
        self._weighted_results_df: None | pd.DataFrame = None
        self._scaling_results: dict = {}
        self._persistent_homology_data: dict[str, None|list|np.ndarray] = {"persistent_homology": None, "point_cloud_data": None}
        self._histogram_data = {"degree_distribution": [0], "clustering_coefficients": [0],
                               "betweenness_distribution": [0], "closeness_distribution": [0],
                               "eigenvector_distribution": [0], "ohms_distribution": [0],
                               "percolation_distribution": [], "weighted_degree_distribution": [0],
                               "weighted_clustering_coefficients": [0], "weighted_betweenness_distribution": [0],
                               "currentflow_distribution": [0], "weighted_closeness_distribution": [0],
                               "weighted_eigenvector_distribution": [0], "weighted_percolation_distribution": [0],}

    @property
    def configs(self) -> dict:
        """Returns the dictionary containing the parameters and options for computing graph theory metrics."""
        return self._configs

    @configs.setter
    def configs(self, configs: dict):
        """Sets the parameters and options for computing graph theory metrics."""
        self._configs = configs

    @property
    def props(self) -> list:
        """Returns the list of properties computed by StructuralGT."""
        return self._props

    @property
    def ntwk_p(self) -> ImageProcessor:
        """Returns the ImageProcessor object."""
        return self._ntwk_p

    @ntwk_p.setter
    def ntwk_p(self, ntwk_p: ImageProcessor):
        """Sets the ImageProcessor object."""
        self._ntwk_p = ntwk_p

    @property
    def plot_figures(self) -> list|None:
        """Returns the list of figures generated by the graph theory metrics computation."""
        return self._plot_figures

    @property
    def results_df(self) -> pd.DataFrame|None:
        """Returns the Pandas DataFrame containing the results of the graph theory metrics computation."""
        return self._results_df

    @property
    def weighted_results_df(self) -> pd.DataFrame|None:
        """Returns the Pandas DataFrame containing the weighted results of the graph theory metrics computation."""
        return self._weighted_results_df

    @property
    def scaling_results(self) -> dict:
        return self._scaling_results

    def track_img_progress(self, status_data: ProgressData) -> None:
        self.update_status(status_data)

    def run_analyzer(self) -> None:
        """
            Execute functions that will process image filters and extract the graph from the grayscale image
        """

        # 1. Get graph extracted from selected images
        graph_obj = self._ntwk_p.graph_obj

        # 2. Apply image filters and extract the graph (only if it has not been executed)
        if graph_obj.nx_giant_graph is None:
            self._ntwk_p.add_listener(self.track_img_progress)
            self._ntwk_p.apply_img_filters()  # Apply image filters
            self._ntwk_p.build_graph_network()  # Extract graph from binary image
            self._ntwk_p.remove_listener(self.track_img_progress)
            self.abort = self._ntwk_p.abort
            if not self.abort:
                self.update_status(ProgressData(percent=100, sender="GT", message=f"Graph successfully extracted!"))
            graph_obj = self._ntwk_p.graph_obj

        if self.abort:
            return

        # 3a. Compute Unweighted GT parameters
        self._results_df = self.compute_gt_metrics(graph_obj.nx_giant_graph)  # replace with graph_obj.nx_giant_graph

        # 3b. Compute Scaling Scatter Plots
        scaling_data = None
        if self._configs["compute_scaling_behavior"]["value"] == 1:
            scaling_data = self.compute_scaling_data()

        if self.abort:
            self.update_status(ProgressData(type="error", sender="GT", message=f"Problem encountered while computing un-weighted GT parameters."))
            return

        # 4. Compute Weighted GT parameters (skip if MultiGraph)
        self._weighted_results_df = self.compute_weighted_gt_metrics(graph_obj)

        if self.abort:
            self.update_status(ProgressData(type="error", sender="GT",
                                            message=f"Problem encountered while computing weighted GT parameters."))
            return

        # 5. Generate results in PDF
        self._plot_figures = self.generate_pdf_output(graph_obj, scaling_data)

        # 6. Save GT compute metrics into props
        self.get_compute_props()

    def compute_gt_metrics(self, graph: nx.Graph|None = None, save_histogram: bool = True, silent: bool = False) -> None|pd.DataFrame:
        """
        Compute unweighted graph theory metrics.

        :param graph: NetworkX graph object.
        :param save_histogram: Whether to save the histogram data.
        :param silent: Whether to send progress status and message (or silence them).

        :return: A Pandas DataFrame containing the unweighted graph theory metrics.
        """

        if graph is None:
            return None

        if not silent:
            self.update_status(ProgressData(percent=1, sender="GT", message=f"Performing un-weighted GT parameters..."))

        opt_gtc = self._configs
        data_dict = {"parameter": [], "value": []}

        node_count = int(nx.number_of_nodes(graph))
        edge_count = int(nx.number_of_edges(graph))

        data_dict["parameter"].append("Number of nodes")
        data_dict["value"].append(node_count)

        data_dict["parameter"].append("Number of edges")
        data_dict["value"].append(edge_count)

        """
        # length of edges
        length_arr = np.array(list(nx.get_edge_attributes(graph, 'length').values()))
        data_dict["parameter"].append('Average length (nm)')
        data_dict["value"].append(round(np.average(length_arr), 3))
        data_dict["parameter"].append('Median length (nm)')
        data_dict["value"].append(round(np.median(length_arr), 3))

        # width of edges
        width_arr = np.array(list(nx.get_edge_attributes(graph, 'width').values()))
        data_dict["parameter"].append('Average width (nm)')
        data_dict["value"].append(round(np.average(width_arr), 3))
        data_dict["parameter"].append('Median width (nm)')
        data_dict["value"].append(round(np.median(width_arr), 3))
        """

        # angle of edges (inbound and outbound)
        angle_arr = np.array(list(nx.get_edge_attributes(graph, 'angle').values()))
        data_dict["parameter"].append('Average edge angle (degrees)')
        data_dict["value"].append(round(np.average(angle_arr), 3))
        data_dict["parameter"].append('Median edge angle (degrees)')
        data_dict["value"].append(round(np.median(angle_arr), 3))

        if graph.number_of_nodes() <= 0:
            if not silent:
                self.update_status(ProgressData(type="error", sender="GT", message=f"Problem with graph (change filter and graph options)."))
            return None

        # creating degree histogram
        if opt_gtc["display_degree_histogram"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=5, sender="GT", message=f"Computing graph degree..."))
            deg_distribution_1 = dict(nx.degree(graph))
            deg_distribution = np.array(list(deg_distribution_1.values()), dtype=float)
            if save_histogram:
                self._histogram_data["degree_distribution"] = deg_distribution
            data_dict["parameter"].append("Average degree")
            data_dict["value"].append(round(np.average(deg_distribution), 5))

        is_connected = False
        if (opt_gtc["compute_network_diameter"]["value"] == 1) or (
                opt_gtc["compute_avg_node_connectivity"]["value"] == 1):
            try:
                is_connected = nx.is_connected(graph)
            except nx.exception.NetworkXPointlessConcept:
                pass

        # calculating network diameter
        if opt_gtc["compute_network_diameter"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=10, sender="GT", message=f"Computing network diameter..."))
            if is_connected:
                dia = int(diameter(graph))
            else:
                dia = np.nan
            data_dict["parameter"].append("Network diameter")
            data_dict["value"].append(dia)

        # calculating average nodal connectivity
        if opt_gtc["compute_avg_node_connectivity"]["value"] == 1:
            if self.abort:
                if not silent:
                    self.update_status(ProgressData(type="error", sender="GT", message=f"Task aborted."))
                return None
            if not silent:
                self.update_status(ProgressData(percent=15, sender="GT", message=f"Computing node connectivity..."))
            avg_node_con = self.compute_avg_node_connectivity(graph, is_connected)
            data_dict["parameter"].append("Average node connectivity")
            data_dict["value"].append(avg_node_con)

        # calculating graph density
        if opt_gtc["compute_graph_density"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=20, sender="GT", message=f"Computing graph density..."))
            g_density = nx.density(graph)
            g_density = round(g_density, 5)
            data_dict["parameter"].append("Graph density")
            data_dict["value"].append(g_density)

        # calculating global efficiency
        if opt_gtc["compute_global_efficiency"]["value"] == 1:
            if self.abort:
                if not silent:
                    self.update_status(ProgressData(type="error", sender="GT", message=f"Task aborted."))
                return None
            if not silent:
                self.update_status(ProgressData(percent=25, sender="GT", message=f"Computing global efficiency..."))
            g_eff = global_efficiency(graph)
            g_eff = round(g_eff, 5)
            data_dict["parameter"].append("Global efficiency")
            data_dict["value"].append(g_eff)

        if opt_gtc["compute_wiener_index"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=30, sender="GT", message=f"Computing Wiener index..."))
            # settings.update_label("Calculating w_index...")
            w_index = wiener_index(graph)
            w_index = round(w_index, 1)
            data_dict["parameter"].append("Wiener Index")
            data_dict["value"].append(w_index)

        # calculating assortativity coefficient
        if opt_gtc["compute_assortativity_coef"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=35, sender="GT", message=f"Computing assortativity coefficient..."))
            a_coef = degree_assortativity_coefficient(graph)
            a_coef = round(a_coef, 5)
            data_dict["parameter"].append("Assortativity coefficient")
            data_dict["value"].append(a_coef)

        # calculating clustering coefficients
        if opt_gtc["compute_avg_clustering_coef"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=40, sender="GT", message=f"Computing average clustering coefficient..."))
            coefficients_1 = clustering(graph)
            if isinstance(coefficients_1, dict):
                cl_coefficients = np.array(list(coefficients_1.values()), dtype=float)
                if save_histogram:
                    self._histogram_data["clustering_coefficients"] = cl_coefficients
                data_dict["parameter"].append("Average clustering coefficient")
                data_dict["value"].append(round(np.average(cl_coefficients), 5))

        # calculating betweenness centrality histogram
        if opt_gtc["display_betweenness_centrality_histogram"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=45, sender="GT", message=f"Computing betweenness centrality..."))
            b_distribution_1 = betweenness_centrality(graph)
            b_distribution = np.array(list(b_distribution_1.values()), dtype=float)
            if save_histogram:
                self._histogram_data["betweenness_distribution"] = b_distribution
            data_dict["parameter"].append("Average betweenness centrality")
            data_dict["value"].append(round(np.average(b_distribution), 5))

        # calculating eigenvector centrality
        if opt_gtc["display_eigenvector_centrality_histogram"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=50, sender="GT", message=f"Computing eigenvector centrality..."))
            try:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=100)
            except nx.exception.PowerIterationFailedConvergence:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=10000)
            e_vecs = np.array(list(e_vecs_1.values()), dtype=float)
            if save_histogram:
                self._histogram_data["eigenvector_distribution"] = e_vecs
            data_dict["parameter"].append("Average eigenvector centrality")
            data_dict["value"].append(round(np.average(e_vecs), 5))

        # calculating closeness centrality
        if opt_gtc["display_closeness_centrality_histogram"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=55, sender="GT", message=f"Computing closeness centrality..."))
            close_distribution_1 = closeness_centrality(graph)
            close_distribution = np.array(list(close_distribution_1.values()), dtype=float)
            if save_histogram:
                self._histogram_data["closeness_distribution"] = close_distribution
            data_dict["parameter"].append("Average closeness centrality")
            data_dict["value"].append(round(np.average(close_distribution), 5))

        # calculating Ohms centrality
        if opt_gtc["display_ohms_histogram"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=60, sender="GT", message=f"Computing Ohms centrality..."))
            o_distribution_1, res = self.compute_ohms_centrality(graph)
            o_distribution = np.array(list(o_distribution_1.values()), dtype=float)
            if save_histogram:
                self._histogram_data["ohms_distribution"] = o_distribution
            data_dict["parameter"].append("Average Ohms centrality")
            data_dict["value"].append(round(np.average(o_distribution), 5))
            data_dict["parameter"].append("Ohms centrality -- avg. area " + r"($m^2$)")
            data_dict["value"].append(round(res['avg area'], 5))
            data_dict["parameter"].append("Ohms centrality -- avg. length (m)")
            data_dict["value"].append(round(res['avg length'], 5))
            data_dict["parameter"].append("Ohms centrality -- avg. width (m)")
            data_dict["value"].append(round(res['avg width'], 5))
            data_dict["parameter"].append("Ohms centrality -- g shape coeff.")
            data_dict["value"].append(round(res['g shape'], 5))
            data_dict["parameter"].append("Ohms centrality -- conductivity (S/m)")
            data_dict["value"].append(round(res['conductivity'], 5))

        # calculating persistent homology
        if opt_gtc["compute_persistent_homology"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=61, sender="GT", message="Computing persistent homology..."))
            ph_dict, mt_dict = self.compute_persistent_homology(graph=graph)
            data_dict["parameter"].append("Persistent Homology--Critical Epsilon")
            data_dict["value"].append(round(mt_dict["critical_epsilon"], 5))
            data_dict["parameter"].append("Persistent Homology--Active Holes Count")
            data_dict["value"].append(mt_dict["active_holes_count"])

            self._persistent_homology_data["persistent_homology"] = ph_dict["persistent_homology"]
            self._persistent_homology_data["point_cloud_data"] = ph_dict["point_cloud_data"]

        return pd.DataFrame(data_dict)

    def compute_weighted_gt_metrics(self, graph_obj: FiberNetworkBuilder|None = None, save_histogram: bool = True, silent: bool = False) -> None|pd.DataFrame:
        """
        Compute weighted graph theory metrics.

        :param graph_obj: GraphExtractor object.
        :param save_histogram: Whether to save histogram data.
        :param silent: Whether to send progress status and message (or silence them).

        :return: A Pandas DataFrame containing the weighted graph theory metrics.
        """
        if graph_obj is None:
            return None

        if not graph_obj.configs["has_weights"]["value"]:
            return None

        if not silent:
            self.update_status(ProgressData(percent=70, sender="GT", message=f"Performing weighted GT parameters..."))

        graph = graph_obj.nx_giant_graph
        opt_gtc = self._configs
        wt_type = graph_obj.get_weight_type()
        weight_type = FiberNetworkBuilder.get_weight_options().get(wt_type)
        data_dict = {"parameter": [], "value": []}

        if graph is None:
            return None

        if graph.number_of_nodes() <= 0:
            if not silent:
                self.update_status(ProgressData(type="error", sender="GT", message=f"Problem with graph (change filter and graph options)."))
            return None

        if opt_gtc["display_degree_histogram"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=72, sender="GT", message=f"Computing weighted degree histogram..."))
            deg_distribution_1 = dict(nx.degree(graph, weight='weight'))
            deg_distribution = np.array(list(deg_distribution_1.values()), dtype=float)
            if save_histogram:
                self._histogram_data["weighted_degree_distribution"] = deg_distribution
            data_dict["parameter"].append(f"{weight_type}-weighted average degree")
            data_dict["value"].append(round(np.average(deg_distribution), 5))

        if opt_gtc["compute_wiener_index"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=74, sender="GT", message=f"Computing weighted Wiener index..."))
            w_index = wiener_index(graph, weight='length')
            w_index = round(w_index, 1)
            data_dict["parameter"].append("Length-weighted Wiener Index")
            data_dict["value"].append(w_index)

        if opt_gtc["compute_avg_node_connectivity"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=76, sender="GT", message=f"Computing weighted average node connectivity..."))
            connected_graph = nx.is_connected(graph)
            if connected_graph:
                max_flow = float(0)
                p = periphery(graph)
                q = len(p) - 1
                for s in range(0, q - 1):
                    for t in range(s + 1, q):
                        flow_value = maximum_flow(graph, p[s], p[t], capacity='weight')[0]
                        if flow_value > max_flow:
                            max_flow = flow_value
                max_flow = round(max_flow, 5)
            else:
                max_flow = np.nan
            data_dict["parameter"].append("Max flow between periphery")
            data_dict["value"].append(max_flow)

        if opt_gtc["compute_assortativity_coef"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=78, sender="GT", message=f"Computing weighted assortativity coefficient..."))
            a_coef = degree_assortativity_coefficient(graph, weight='width')
            a_coef = round(a_coef, 5)
            data_dict["parameter"].append("Width-weighted assortativity coefficient")
            data_dict["value"].append(a_coef)

        if opt_gtc["display_betweenness_centrality_histogram"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=80, sender="GT", message=f"Computing weighted betweenness centrality..."))
            b_distribution_1 = betweenness_centrality(graph, weight='weight')
            b_distribution = np.array(list(b_distribution_1.values()), dtype=float)
            if save_histogram:
                self._histogram_data["weighted_betweenness_distribution"] = b_distribution
            data_dict["parameter"].append(f"{weight_type}-weighted betweenness centrality")
            data_dict["value"].append(round(np.average(b_distribution), 5))

        if opt_gtc["display_closeness_centrality_histogram"]["value"] == 1:
            if not silent:
                self.update_status(ProgressData(percent=82, sender="GT", message=f"Computing weighted closeness centrality..."))
            close_distribution_1 = closeness_centrality(graph, distance='length')
            close_distribution = np.array(list(close_distribution_1.values()), dtype=float)
            if save_histogram:
                self._histogram_data["weighted_closeness_distribution"] = close_distribution
            data_dict["parameter"].append(f"Length-weighted average closeness centrality")
            data_dict["value"].append(round(np.average(close_distribution), 5))

        if opt_gtc["display_eigenvector_centrality_histogram"]["value"] == 1:
            if self.abort:
                if not silent:
                    self.update_status(ProgressData(type="error", sender="GT", message=f"Task aborted."))
                return None
            if not silent:
                self.update_status(ProgressData(percent=84, sender="GT", message=f"Computing weighted eigenvector centrality..."))
            try:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=100, weight='weight')
            except nx.exception.PowerIterationFailedConvergence:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=10000, weight='weight')
            e_vecs = np.array(list(e_vecs_1.values()), dtype=float)
            if save_histogram:
                self._histogram_data["weighted_eigenvector_distribution"] = e_vecs
            data_dict["parameter"].append(f"{weight_type}-weighted average eigenvector centrality")
            data_dict["value"].append(round(np.average(e_vecs), 5))

        # calculate cross-sectional area of edges
        wt_type = graph_obj.get_weight_type()
        if wt_type == 'AREA':
            if not silent:
                self.update_status(ProgressData(percent=88, sender="GT", message=f"Computing weighted average (edge) cross-sectional area..."))
            temp_distribution = []
            for (s, e) in graph.edges():
                temp_distribution.append(graph[s][e]['weight'])
            a_distribution = np.array(temp_distribution, dtype=float)
            ae_val = np.average(a_distribution)
            ae_val = round(ae_val, 5)
            data_dict["parameter"].append(f"Average edge cross-sectional area (nm\u00b2)")
            data_dict["value"].append(ae_val)

        return pd.DataFrame(data_dict)

    def compute_persistent_homology(self, graph: nx.Graph | None = None, silent: bool = False) -> tuple[dict[str, list|np.ndarray|None], dict[str, float]]:
        """
        Persistent homology is a method in TDA (Topological Data Analysis) that tries to identify topological features
        (i.e., holes, connected components) which do not change in point cloud data during construction of various
        simplicial complexes (filtration).

        To compute persistent homology by growing circles around nodes, you are building a Vietoris–Rips filtration
        on the point cloud. As the radius (distance) increases, circles overlap, edges form, and when three nodes
        are mutually connected, a triangle (2-simplex) is filled in to accurately measure the birth and death of holes.

        In this method, we compute the persistent homology by converting a NetworkX graph into a point cloud data
        using its nodes.

        Next, we extract scale-based topological metrics from persistent homology data. This function analyzes
        persistence_data intervals from a filtration and computes:

        1. Critical cluster threshold (`critical_epsilon`)
            The largest finite death value in dimension 0 (`H0`), corresponding
            to the filtration scale at which the final two connected components
            merge into a single connected component.

        2. Active holes count (`active_holes_count`)
            The number of dimension 1 (`H1`) features (loops/holes) that are
           alive at the critical scale, i.e. features satisfying:

               birth <= critical_epsilon < death

        :param graph: NetworkX graph object.
        :param silent: Whether to send progress status and message (or silence them).
        """

        def extract_scale_metrics(ph_data):
            """
            Extract scale-based topological metrics from persistent homology data. This function analyzes persistence_data
            intervals from a filtration and computes:

            1. Critical cluster threshold (`critical_epsilon`)
               The largest finite death value in dimension 0 (`H0`), corresponding
               to the filtration scale at which the final two connected components
               merge into a single connected component.

            2. Active holes count (`active_holes_count`)
               The number of dimension 1 (`H1`) features (loops/holes) that are
               alive at the critical scale, i.e. features satisfying:

                   birth <= critical_epsilon < death

            Parameters
            ----------
            ph_data : iterable of tuple
                Persistent homology intervals in the form:

                    [(dimension, (birth, death)), ...]

                where:
                - dimension : int
                    Homology dimension (`0` for connected components,
                    `1` for loops, etc.)
                - birth : float
                    Filtration value at which the feature appears.
                - death : float
                    Filtration value at which the feature disappears.
                    Infinite values are allowed.

            Returns
            -------
            dict
                Dictionary containing:
                - "critical_epsilon" : float
                    Largest finite `H0` death value.
                - "active_holes_count" : int
                    Number of `H1` features alive at `critical_epsilon`.

            Raises
            ------
            ValueError
                If no finite `H0` death values are present.

            Notes
            -----
            The surviving connected component in `H0` typically has death = inf
            and is excluded from the computation.
            """

            # 1. Extract all finite death values for Dimension 0 (Connected Components)
            h0_deaths = [
                death for dim, (birth, death) in ph_data
                if dim == 0 and death != float('inf') and np.isfinite(death)
            ]

            if not h0_deaths:
                raise ValueError("No finite H0 death values found in the data.")

            # The last epsilon value that reduces the graph to 1 component
            critical_epsilon = max(h0_deaths)

            # 2. Count the number of Dimension 1 holes alive at this critical scale
            # A hole is alive if: Birth <= critical_epsilon < Death
            active_holes_count = sum(
                1 for dim, (birth, death) in ph_data
                if dim == 1 and birth <= critical_epsilon < death
            )

            return {
                "critical_epsilon": critical_epsilon,
                "active_holes_count": active_holes_count
            }

        res_dict: dict[str, list|np.ndarray|None] = { "persistent_homology": None, "point_cloud_data": None}
        metric_dict: dict[str, float] = { "critical_epsilon": 0.0, "active_holes_count": 0.0 }

        if graph is None:
            return res_dict, metric_dict

        if not silent:
            self.update_status(ProgressData(type='info', sender="GT", message="Computing persistent homology..."))

        # 1. Safely extract positions to form pure point cloud data from nodes
        try:
            node_list = list(graph.nodes())
            pos = {i: graph.nodes[i]['o'] for i in node_list}

            # Convert node coordinates into a standard NumPy array (Pure Point Cloud)
            sorted_nodes = sorted(graph.nodes())
            point_cloud = np.array([pos[node] for node in sorted_nodes])
        except nx.exception.NetworkXNoPath:
            # if position-data 'o' does not exist
            return res_dict, metric_dict

        # 2. Build the Rips Complex (growing circles around pure node data).
        try:
            # This completely ignores existing graph edges and generates its own complexes/triangles
            rips_complex = gudhi.RipsComplex(points=point_cloud)

            # Create the simplex tree up to dimension 2 (0 = components, 1 = holes, 2 = filled triangles)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)  # Dimension 2: Triangles (Solid faces connecting 3 points)

            # 3. Compute Persistent Homology
            persistence_data = simplex_tree.persistence()
        except exception:
            return res_dict, metric_dict

        # 3. Compute persistence metrics
        metric_dict = extract_scale_metrics(persistence_data)
        res_dict["persistent_homology"] = persistence_data
        res_dict["point_cloud_data"] = point_cloud
        return res_dict, metric_dict

    def compute_scaling_data(self) -> defaultdict:
        """
        Iteratively divides the input image into smaller windows (filters), extracts graphs from each window,
        and computes their corresponding ground truth (GT) parameters.

        The method aggregates these parameters across multiple filter sizes to analyze scaling behavior,
        storing the results in a dictionary for further processing or visualization.
        """

        self.update_status(ProgressData(percent=65, sender="GT", message=f"Computing scaling behaviour..."))
        self._ntwk_p.add_listener(self.track_img_progress)
        num_filters = int(self._configs["scaling_behavior_kernel_count"]["value"])
        num_patches = int(self._configs["scaling_behavior_patches_per_kernel"]["value"])
        calc_avg = self._configs["scaling_behavior_compute_avg"]["value"]
        graph_groups = self._ntwk_p.build_graph_from_patches(num_kernels=num_filters, patch_count_per_kernel=num_patches, compute_avg=calc_avg)
        self._ntwk_p.remove_listener(self.track_img_progress)

        sorted_plt_data = defaultdict(lambda: defaultdict(list))
        avg_df = None
        for (h, w), nx_graphs in graph_groups.items():
            num_graphs = len(nx_graphs)
            for i, nx_graph in enumerate(nx_graphs):
                self.update_status(ProgressData(type="warning", sender="GT", message=f"Computing GT parameters for filter {h}x{w}: graph-patch {i + 1}/{num_graphs}..."))
                temp_df = self.compute_gt_metrics(nx_graph, save_histogram=False, silent=True)
                if temp_df is None:
                    # Skip the problematic graph
                    continue
                for _, row in temp_df.iterrows():
                    x_param = row["parameter"]
                    y_value = row["value"]
                    if ' edge angle' in x_param:  # Skip this
                        continue
                    sorted_plt_data[x_param][h].append(y_value) if num_graphs > 4 else None

                # Save GT parameters/descriptors of 90% image to DF
                if num_graphs > 4:
                    continue
                else:
                    temp_df = temp_df.rename(columns={'value': f'value-{i + 1}'})
                    if i == 0:
                        avg_df = temp_df
                    else:
                        avg_df = avg_df.merge(temp_df, on='parameter') if avg_df is not None else temp_df

        # Add average to scaling results (for the Excel file)
        if avg_df is not None:
            self._scaling_results["SGT Descriptors"] = avg_df

        return sorted_plt_data

    def compute_ohms_centrality(self, nx_graph: nx.Graph) -> tuple[dict, dict] | tuple[None, None]:
        r"""
        Computes Ohms centrality value for each node based on actual pixel width and length of edges in meters.

        :param nx_graph: NetworkX graph object.

        Returns: Ohms centrality distribution
        """
        ohms_dict = {}
        lst_area = []
        lst_len = []
        lst_width = []

        if self._ntwk_p.selected_batch.is_graph_only:
            return None, None
        sel_images = self._ntwk_p.selected_images
        px_sizes = np.array([img.configs["pixel_width"]["value"] for img in sel_images])
        rho_dims = np.array([img.configs["resistivity"]["value"] for img in sel_images])

        px_size = float(np.average(px_sizes.astype(float)))
        rho_dim = float(np.average(rho_dims.astype(float)))
        pixel_dim = px_size  # * (10 ** 9)  # Convert to nanometers
        g_shape = 1

        b_dict = betweenness_centrality(nx_graph)
        lst_nodes = list(nx_graph.nodes())
        for n in lst_nodes:
            # compute Ohms centrality value for each node
            b_val = float(b_dict[n])
            if b_val == 0:
                ohms_val = 0
            else:
                connected_nodes = nx_graph[n]  # all nodes connected to node n
                arr_len = []
                arr_dia = []
                for idx, val in connected_nodes.items():
                    arr_len.append(val['length'])
                    arr_dia.append(val['width'])
                arr_len = np.array(arr_len, dtype=float)
                arr_dia = np.array(arr_dia, dtype=float)

                pix_width = float(np.average(arr_dia))
                pix_length = np.sum(arr_len)
                length = pix_length * pixel_dim
                width = pix_width * pixel_dim
                # area = math.pi * 89.6 * (width * 0.5) ** 2
                area = g_shape * (width * width)
                ohms_val = ((b_val * length * rho_dim) / area)
                lst_len.append(length)
                lst_area.append(area)
                lst_width.append(width)
                # if n < 5:
            ohms_dict[n] = ohms_val
        avg_area = np.average(np.array(lst_area, dtype=float))
        med_area = np.median(np.array(lst_area, dtype=float))
        avg_len = np.average(np.array(lst_len, dtype=float))
        med_len = np.median(np.array(lst_len, dtype=float))
        avg_width = np.average(np.array(lst_width, dtype=float))
        med_width = np.median(np.array(lst_width, dtype=float))
        res = {
            'avg area': avg_area, 'med area': med_area,
            'avg length': avg_len, 'med length': med_len,
            'avg width': avg_width, 'med width': med_width,
            'g shape': g_shape, 'conductivity': (1 / rho_dim)}

        return ohms_dict, res

    def compute_avg_node_connectivity(self, nx_graph: nx.Graph, is_graph_connected=False) -> float:
        r"""Returns the average connectivity of a graph G.

        The average connectivity `\bar{\kappa}` of a graph G is the average
        of local node connectivity over all pairs of the nx_graph nodes.

        :param nx_graph: NetworkX graph object.
        :param is_graph_connected: Boolean
        """

        def igraph_average_node_connectivity():
            r"""
            Returns the average connectivity of a graph G.

            The average connectivity of a graph G is the average
            of local node connectivity over all pairs of the Graph (G) nodes.
            """

            ig_graph = ig.Graph.from_networkx(nx_graph)
            num_nodes = ig_graph.vcount()
            total, count = 0, 0
            with multiprocessing.Pool() as pool:
                # Prepare all node pairs (i < j)
                items = [(ig_graph, i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
                async_result = pool.starmap_async(_worker_vertex_connectivity, items)
                for n in async_result.get():
                    if self.abort:
                        self.update_status(ProgressData(type="error", sender="GT", message=f"Task aborted."))
                        pool.terminate()
                        pool.join()
                        return 0
                    if n is not None:
                        total += n
                        count += 1
            anc = total / count if count > 0 else 0
            return anc

        def igraph_clang_average_node_connectivity():
            r"""Returns the average connectivity of a graph G.

            The average connectivity of a graph G is the average
            of local node connectivity over all pairs of the Graph (G) nodes.

            """
            from .c_lang import sgt_c_module as sgt

            cpu_count = get_num_cores()
            num_threads = cpu_count if nx.number_of_nodes(nx_graph) < 2000 else cpu_count * 2
            anc = 0

            try:
                filename, output_location = self._ntwk_p.get_filenames()
                g_filename = filename + "_graph.txt"
                graph_file = os.path.join(output_location, g_filename)
                nx.write_edgelist(nx_graph, graph_file, data=False)
                anc = sgt.compute_anc(graph_file, num_threads, self._allow_mp)
            except Exception as err:
                logging.exception("Computing ANC Error: %s", err, extra={'user': 'SGT Logs'})
            return anc

        if is_graph_connected:
            # use_igraph = opt_gtc["computing_lang == 'C'"]["value"]
            if self._use_igraph:
                # use iGraph Lib in C
                self.update_status(ProgressData(percent=15, sender="GT", message=f"Using iGraph library..."))
                try:
                    avg_node_con = igraph_clang_average_node_connectivity()
                except ImportError:
                    avg_node_con = igraph_average_node_connectivity()
            else:
                # Use NetworkX Lib in Python
                self.update_status(ProgressData(percent=15, sender="GT", message=f"Using NetworkX library..."))
                avg_node_con = average_node_connectivity(nx_graph)
            avg_node_con = round(avg_node_con, 5)
        else:
            avg_node_con = np.nan
        return avg_node_con

    def compute_graph_conductance(self, graph_obj):
        """
        Computes graph conductance through an approach based on eigenvectors or spectral frequency.
        Implements ideas proposed in:    https://doi.org/10.1016/j.procs.2013.09.311.

        Conductance can closely be approximated via eigenvalue computation,
        a fact which has been well-known and well-used in the graph theory community.

        The Laplacian matrix of a directed graph is by definition generally non-symmetric,
        while, e.g., traditional spectral clustering is primarily developed for undirected
        graphs with symmetric adjacency and Laplacian matrices. A trivial approach to applying the
        techniques requiring symmetry is to turn the original directed graph into an
        undirected graph and build the Laplacian matrix for the latter.

        We need to remove isolated nodes (to avoid singular adjacency matrix).
        The degree of a node is the number of edges incident to that node.
        When a node has a degree of zero, it means that there are no edges
        connected to that node. In other words, the node is isolated from
        the rest of the graph.

        :param graph_obj: Graph Extractor object.

        """
        self.update_status(ProgressData(type="warning", sender="GT", message=f"Computing graph conductance..."))
        # Make a copy of the graph
        graph = graph_obj.nx_giant_graph.copy()
        weighted = graph_obj.configs["has_weights"]["value"]

        # It is important to notice our graph is (mostly) a directed graph,
        # meaning that it is: (asymmetric) with self-looping nodes

        # 1. Remove self-looping edges from the graph, they cause zero values in Degree matrix.
        # 1a. Get Adjacency matrix
        adj_mat = nx.adjacency_matrix(graph).todense()

        # 1b. Remove (self-loops) non-zero diagonal values in Adjacency matrix
        np.fill_diagonal(adj_mat, 0)

        # 1c. Create the new graph
        giant_graph = nx.from_numpy_array(adj_mat)

        # 2a. Identify isolated nodes
        isolated_nodes = list(nx.isolates(giant_graph))

        # 2b. Remove isolated nodes
        giant_graph.remove_nodes_from(isolated_nodes)

        # 3a. Check the connectivity of the graph
        # It has less than two nodes or is not connected.
        # Identify connected components
        connected_components = list(nx.connected_components(graph))
        if not connected_components:  # In case the graph is empty
            connected_components = []
        sub_graphs = [graph.subgraph(c).copy() for c in connected_components]

        giant_graph = max(sub_graphs, key=lambda g: g.number_of_nodes())

        # 4. Compute normalized-laplacian matrix
        if weighted:
            norm_laplacian_matrix = nx.normalized_laplacian_matrix(giant_graph, weight='weight').toarray()
        else:
            # norm_laplacian_matrix = compute_norm_laplacian_matrix(giant_graph)
            norm_laplacian_matrix = nx.normalized_laplacian_matrix(giant_graph).toarray()

        # 5. Compute eigenvalues
        # e_vals, _ = xp.linalg.eig(norm_laplacian_matrix)
        e_vals = sp.linalg.eigvals(norm_laplacian_matrix)

        # 6. Approximate conductance using the 2nd smallest eigenvalue
        # 6a. Compute the minimum and maximum values of graph conductance.
        sorted_vals = np.array(e_vals.real)
        sorted_vals.sort()
        # approximate conductance using the 2nd smallest eigenvalue
        try:
            # Maximum Conductance
            val_max = math.sqrt((2 * sorted_vals[1]))
        except ValueError:
            val_max = np.nan
        # Minimum Graph Conductance
        val_min = sorted_vals[1] / 2

        return val_max, val_min

    def get_config_info(self) -> str:
        """
        Get the user selected parameters and options information.
        :return:
        """

        opt_gtc = self._configs
        run_info = ""
        parts = []

        if opt_gtc["compute_scaling_behavior"]["value"] == 1:
            num_filters = int(opt_gtc["scaling_behavior_kernel_count"]["value"])
            num_patches = int(opt_gtc["scaling_behavior_patches_per_kernel"]["value"])

            parts.append(f"Kernel Count = {num_filters}")
            parts.append(f"No. of Random Locations = {num_patches}")

        # Add title if needed
        if parts:
            # Join cleanly without worrying about leftover separators
            run_info = " || ".join(parts)
            run_info = f"***Graph Computation Configurations***\n{run_info}"

        return run_info

    def get_compute_props(self) -> None:
        """
        A method that retrieves graph theory computed parameters and stores them in a list-array.

        Returns: list of computed GT params.

        """
        self._props = []
        # 1. Unweighted parameters
        if self._results_df is None:
            return
        param_df = self._results_df.copy()
        self._props.append(['UN-WEIGHTED', 'PARAMETERS'])
        for _, row in param_df.iterrows():
            x_param = row["parameter"]
            y_value = row["value"]
            self._props.append([x_param, y_value])

        # 2. Weighted parameters
        if self._weighted_results_df is None:
            return
        param_df = self._weighted_results_df.copy()
        self._props.append(['WEIGHTED', 'PARAMETERS'])
        for _, row in param_df.iterrows():
            x_param = row["parameter"]
            y_value = row["value"]
            self._props.append([x_param, y_value])

    def generate_pdf_output(self, graph_obj: FiberNetworkBuilder, scaling_data=None) -> list[plt.Figure]:
        """
        Generate results as graphs and plots which should be written in a PDF file.

        :param graph_obj: Graph extractor object.
        :param scaling_data: Computed scaling results stored in a dictionary.

        :return: List of results.
        """

        self.update_status(ProgressData(percent=90, sender="GT", message=f"Writing results to PDF..."))
        opt_gtc = self._configs
        out_figs = []

        sel_images = self._ntwk_p.selected_images
        img_3d = np.asarray(self._ntwk_p.image_3d) if len(sel_images) > 0 else None

        def plot_gt_results():
            """
            Create a table of weighted and unweighted graph theory results.

            :return: Matplotlib figures of unweighted and weighted graph theory results.
            """

            opt_gte = graph_obj.configs
            data = self._results_df
            w_data = self._weighted_results_df
            col_width = [2 / 3, 1 / 3]

            if data is not None:
                plt_fig = plt.Figure(figsize=(8.5, 11), dpi=300)
                ax = plt_fig.add_subplot(1, 1, 1)
                ax.set_axis_off()
                ax.set_title("Unweighted GT parameters")
                tab_1 = tbl.table(ax, cellText=data.values[:, :], loc='upper center', colWidths=col_width, cellLoc='left')
                tab_1.scale(1, 1.5)
            else:
                plt_fig = None

            if opt_gte is None:
                return plt_fig, None

            if opt_gte["has_weights"]["value"] == 1 and w_data is not None:
                plt_fig_wt = plt.Figure(figsize=(8.5, 11), dpi=300)
                ax = plt_fig_wt.add_subplot(1, 1, 1)
                ax.set_axis_off()
                ax.set_title("Weighted GT parameters")
                tab_2 = tbl.table(ax, cellText=w_data.values[:, :], loc='upper center', colWidths=col_width,
                                  cellLoc='left')
                tab_2.scale(1, 1.5)
            else:
                plt_fig_wt = None
            return plt_fig, plt_fig_wt

        def plot_bin_images():
            """
            Create plot figures of original, grayscale, and binary image.

            :return:
            """

            plt_figs = []
            is_3d = True if len(sel_images) > 1 else False

            for i, img in enumerate(sel_images):
                raw_img = img.img_2d
                img_gray = img.img_grayscale
                img_bin = img.img_bin

                plt_fig = plt.Figure(figsize=(8.5, 8.5), dpi=400)
                ax_1 = plt_fig.add_subplot(2, 2, 1)
                ax_2 = plt_fig.add_subplot(2, 2, 2)
                ax_3 = plt_fig.add_subplot(2, 2, 3)
                ax_4 = plt_fig.add_subplot(2, 2, 4)

                ax_1.set_title(f"Frame {i}: Original Image") if is_3d else ax_1.set_title(f"Original Image")
                ax_1.set_axis_off()
                ax_1.imshow(raw_img, cmap='gray')

                ax_2.set_title(f"Frame {i}: Grayscale Image") if is_3d else ax_2.set_title(f"Grayscale Image")
                ax_2.set_axis_off()
                ax_2.imshow(img_gray, cmap='gray')

                ax_3.set_title(f"Frame {i}: Binary Image") if is_3d else ax_3.set_title(f"Binary Image")
                ax_3.set_axis_off()
                ax_3.imshow(img_bin, cmap='gray')

                img.plot_img_histogram(axes=ax_4)
                hist_title = f"Frame {i}: Histogram of Grayscale Image" if is_3d else f"Histogram of Grayscale Image"
                ax_4.set_title(hist_title)

                plt_figs.append(plt_fig)
            return plt_figs

        def plot_run_configs():
            """
            Create a page (as a figure) that will show the user-selected parameters and options.

            :return: A Matplotlib figure object.
            """

            plt_fig = plt.Figure(figsize=(8.5, 8.5), dpi=300)
            ax = plt_fig.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.set_title("Run Info")

            # similar to the start of the csv file, this is just getting all the relevant settings to display in the PDF
            _, filename = os.path.split(self._ntwk_p.img_path)
            now = datetime.datetime.now()

            run_info = ""
            run_info += filename + "\n"
            run_info += now.strftime("%Y-%m-%d %H:%M:%S") + "\n----------------------------\n\n"

            # Image Configs
            run_info += self._ntwk_p.image_obj.get_config_info()  # Get configs of first image
            run_info += "\n\n"

            # Graph Configs
            run_info += graph_obj.get_config_info()
            run_info += "\n\n"

            # Computation Configs
            run_info += self.get_config_info()
            run_info += "\n\n"

            ax.text(0.5, 0.5, run_info, horizontalalignment='center', verticalalignment='center')
            return plt_fig

        def plot_scaling_behavior():
            """"""

            def plot_axis(subplot_num, plt_type="", plot_err=True):
                """"""
                subplot_num += 1
                axis = plt_fig.add_subplot(2, 2, subplot_num)
                if plot_err:
                    axis.errorbar(x_avg, y_avg, xerr=x_err, yerr=y_err, label='Data', color='b', capsize=4, marker='s',
                                  markersize=4, linewidth=1, linestyle='-')
                axis.set_title(f"{plt_type}\nNodes vs {y_title}", fontsize=10)
                axis.set(xlabel='No. of Nodes', ylabel=f'{param_name}')
                # axis.legend()
                return axis, subplot_num

            # Initialize plot figures
            plt_figs = []
            if scaling_data is None:
                return plt_figs

            # Plot scaling behavior
            self.update_status(ProgressData(percent=91, sender="GT", message=f"Plotting scaling behavior..."))
            i = 0
            x_label, best_scale = None, None
            y_title = ""
            x_values, x_avg, x_err, x_fit = None, None, None, None
            data_df, fit_data_df = None, None
            plt_fig = plt.Figure(figsize=(8.5, 11), dpi=300)
            for param_name, plt_dict in scaling_data.items():
                # axis title name
                y_title = param_name.split('(')[0] if '(' in param_name else param_name

                # Retrieve plot data
                kernel_dims = np.array(sorted(plt_dict.keys()))  # Optional: sort heights and save as a numpy array
                y_lst = [plt_dict[d] for d in kernel_dims]  # shape: (n_samples, n_kernels)

                # Pad with NaN
                max_len = max(len(row) for row in y_lst)
                padded_lst = [row + [np.nan] * (max_len - len(row)) for row in y_lst]

                # Convert to a Numpy array
                y_values: np.ndarray = np.array(padded_lst).T
                y_avg: np.ndarray = np.nanmean(y_values, axis=0)
                y_err: np.ndarray = np.nanstd(y_values, axis=0, ddof=1) / np.sqrt(y_values.shape[0])
                if np.any(np.isnan(y_avg)):
                    # print(f"{param_name} has NaN values: {y_avg}")
                    continue

                # Plot (taking Node-count as the independent variable) against other parameters
                add_plot = False
                if x_label is None:
                    # First Param becomes X-axis: 'Number of Nodes'
                    x_label = param_name
                    # x_values = y_values
                    x_avg: np.ndarray = y_avg
                    x_err: np.ndarray = y_err
                    x_fit: np.ndarray = np.linspace(min(x_avg), max(x_avg), 100)

                    # Estimate best scale
                    # best_scale = find_elbow(kernel_dims[:-1], y_avg[:-1])

                    # Add plots to Figure
                    try:
                        # a) Plot Nodes vs. Kernel-size (scatter plot)
                        # Plot for each Parameter?
                        ax, i = plot_axis(i, "", plot_err=False)
                        ax.set_title(f"Kernel Size vs {y_title}", fontsize=10)
                        ax.set(xlabel='Kernel Size', ylabel=y_title)
                        ax.errorbar(kernel_dims, y_avg, yerr=y_err, label='Data', color='b', capsize=4, marker='s',
                                    markersize=4, linewidth=1, linestyle='-')

                        # b) Log-log plot of Kernel-size vs. Node-count
                        log_x = np.log10(kernel_dims)
                        log_y = np.log10(y_avg)
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(log_x, log_y)
                        log_y_fit = slope * log_x + intercept  # Compute line of best-fit

                        ax = plt_fig.add_subplot(2, 2, i+1)
                        ax.set_title(f"Log-Log Plot of\nKernel Size vs {y_title}", fontsize=10)
                        ax.set(xlabel='Kernel Size', ylabel=y_title)
                        ax.plot(log_x, log_y, label='Data', color='b', marker='s', markersize=3)
                        ax.plot(log_x, log_y_fit, label=f'Fit: slope={slope:.2f}, $R^2$={r_value ** 2:.3f}', color='r')
                        ax.legend()

                        plt_figs.append(plt_fig)
                    except Exception as e:
                        logging.exception("Scaling Law (Kernel-Nodes) Error: %s", e, extra={'user': 'SGT Logs'})

                    # Write to DataFrame
                    data_df = pd.DataFrame({'kernel-dim': kernel_dims, 'x-avg': x_avg, 'x-std': x_err})
                    fit_data_df = pd.DataFrame({'x-fit': x_fit})

                    # Add Kernel DataFrame (with BestScale at the last row)
                    kernel_df = pd.DataFrame({'kernel-dim': kernel_dims, 'x-avg': x_avg, 'x-std': x_err})
                    # new_row = pd.DataFrame([{'kernel-dim': best_scale, 'x-avg': 0.0, 'x-std': 0.0}])
                    # kernel_df = pd.concat([kernel_df, new_row], ignore_index=True)  # Add as last row
                    self._scaling_results["Nodes-Kernel Size"] = kernel_df.copy()
                else:
                    # 2a. Plot on the Log-Log scale
                    try:
                        # 1. Transform to log-log scale
                        log_x = np.log10(x_avg)
                        log_y = np.log10(y_avg)

                        # 2. Perform linear regression in log-log scale
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(log_x, log_y)
                        log_y_fit = slope * log_x + intercept  # Compute line of best-fit

                        # 3. Plot data (Log-Log scale with the line best-fit)
                        ax, i = plot_axis(i, "Log-Log Plot of", plot_err=False)
                        ax.plot(log_x, log_y, label='Data', color='b', marker='s', markersize=3)
                        ax.plot(log_x, log_y_fit, label=f'Fit: slope={slope:.2f}, $R^2$={r_value ** 2:.3f}', color='r')
                        ax.legend()
                        add_plot = True

                        # Write to DataFrame
                        if data_df is not None:
                            data_df['y-avg'] = y_avg
                            data_df['y-std'] = y_err
                            data_df['log-x'] = log_x
                            data_df['log-y'] = log_y
                            data_df['log-y-fit'] = log_y_fit
                    except Exception as err:
                        add_plot = False
                        logging.exception("Scaling Law (Log-Log Fit) Error: %s", err, extra={'user': 'SGT Logs'})

                    if opt_gtc["scaling_behavior_power_law_fit"]["value"] == 1:
                        # 2b. Compute the line of best-fit on our data according to our power-law model
                        try:
                            # Generate points for the best-fit curve
                            y_fit_pwr, params = CurveFitModels.power_law(x_avg, y_avg, x_fit)
                            a_fit, k_fit = params["a"], params["k"]

                            # Compute Kolmogorov-Smirnov Test & Goodness-of-fit P-Values
                            res_good_fit = sp.stats.goodness_of_fit(sp.stats.powerlaw, y_avg)
                            ks_stat, ks_p_val = res_good_fit.statistic, res_good_fit.pvalue

                            # 3b. Plot data (power-law best fit)
                            ax, i = plot_axis(i, "Power Law Fit and Plot of")
                            ax.plot(x_fit, y_fit_pwr,
                                    label=f'Fit: $y = a \\cdot x^{{-k}}$\n$a={a_fit:.2f}, k={k_fit:.2f}$\nKS Stat={ks_stat:.2f}, P-Val={ks_p_val:.2f}',
                                    color='red')
                            ax.legend(fontsize=6)

                            # Write to DataFrame
                            if fit_data_df is not None:
                                fit_data_df['Pwr. Law y-fit'] = y_fit_pwr
                        except Exception as err:
                            logging.exception("Scaling Law (Power Law Fit) Error: %s", err, extra={'user': 'SGT Logs'})

                    if opt_gtc["scaling_behavior_stretched_power_law_fit"]["value"] == 1:
                        # 2c. Compute the line of best-fit according to our stretched power-law model
                        try:
                            # Generate points for the best-fit curve
                            y_fit_cut, params = CurveFitModels.stretched_power_law(x_avg, y_avg, x_fit)
                            a_fit, k_fit, cut_fit, beta_fit = params["a"], params["k"], params["x_c"], params["beta"]

                            # 3c. Plot data (stretched power-law best fit)
                            ax, i = plot_axis(i, "Power Law (w. Exponential Cutoff) Fit")
                            ax.plot(x_fit, y_fit_cut,
                                    label=f"Fit: $y = a \\cdot x^{{-k}} \\cdot \\exp(-(x / x_c)^\\beta)$\n$a={a_fit:.2f}, k={k_fit:.2f}, x_c={cut_fit:.2f}, \\beta={beta_fit:.2f}$",
                                    color='red')
                            ax.legend(fontsize=6)

                            # Write to DataFrame
                            if fit_data_df is not None:
                                fit_data_df['Trunc. Pwr. Law y-fit'] = y_fit_cut
                        except Exception as err:
                            logging.exception("Scaling Law (Stretched Power Law Fit) Error: %s", err,
                                              extra={'user': 'SGT Logs'})

                    if opt_gtc["scaling_behavior_log_normal_fit"]["value"] == 1:
                        # 2d. Compute best-fit, assuming Log-Normal dependence on X
                        try:
                            # Generate predicted points for the best-fit curve
                            y_fit_ln, params = CurveFitModels.lognormal(x_avg, y_avg, x_fit)
                            mu_fit, sigma_fit, a_log_fit = params["mu"], params["sigma"], params["a"]

                            # Compute Kolmogorov-Smirnov Test & Goodness-of-fit P-Values
                            res_good_fit = sp.stats.goodness_of_fit(sp.stats.lognorm, y_avg)
                            ks_stat, ks_p_val = res_good_fit.statistic, res_good_fit.pvalue

                            # 3c. Plot data (Log-normal distribution best fit)
                            ax, i = plot_axis(i, "Log-Normal Fit and Plot of")
                            ax.plot(x_fit, y_fit_ln,
                                    label=f'Fit: log-normal shape\n$\\mu={mu_fit:.2f}$, $\\sigma={sigma_fit:.2f}$\nKS Stat={ks_stat:.2f}, P-Val={ks_p_val:.2f}',
                                    color='red')
                            ax.legend(fontsize=6)

                            # Write to DataFrame
                            if fit_data_df is not None:
                                fit_data_df['Log-Normal y-fit'] = y_fit_ln
                        except Exception as err:
                            logging.exception("Scaling Law (Log-Normal Fit) Error: %s", err, extra={'user': 'SGT Logs'})

                # Navigate to the next subplot
                if (i + 1) > 1:
                    if add_plot:
                        plt_figs.append(plt_fig)
                        df_title = f"Nodes-{y_title}"
                        df_title = f"{df_title[:20]}." if len(df_title) > 20 else df_title
                        self._scaling_results[df_title] = data_df.copy() if data_df is not None else {}
                        self._scaling_results[f"{df_title} (Fitting)"] = fit_data_df.copy() if fit_data_df is not None else {}
                    plt_fig = plt.Figure(figsize=(8.5, 11), dpi=300)
                    i = 0

            plt_figs.append(plt_fig) if i <= 4 else None

            if best_scale is not None:
                new_row = {'parameter': 'Optimal Image Scale (px)', 'value': int(str(best_scale)) if best_scale is not None else 0}
                self._results_df = pd.concat([self._results_df, pd.DataFrame([new_row])], ignore_index=True)

            return plt_figs

        def plot_persistent_homology():
            """
            Custom persistence diagram plotter.
            """
            self.update_status(ProgressData(percent=92, sender="GT", message=f"Plotting persistence diagrams..."))

            def plot_persistence_diagram(ax: plt.Axes | None = None):

                if ph_data is None:
                    return

                if ax is None:
                    ph_fig, ax = plt.subplots(figsize=(6, 6))

                # Default labels
                custom_labels = {0: r"connected components ($\beta_0$)", 1: r"holes ($\beta_1$)"}

                # Matplotlib default color cycle (stable & familiar)
                colors = ['r', 'b', 'g', 'c', 'm', 'y']

                # Group points by dimension
                group_infinite = {}
                group_finite = {}
                max_val = 0
                for dim, (b, d) in ph_data:
                    if d == float("inf") or np.isinf(d):
                        group_infinite.setdefault(dim, []).append(b)
                    else:
                        group_finite.setdefault(dim, []).append((b, d))
                        max_val = max(max_val, b, d)

                # Set infinity line if not provided
                min_lim = -(max_val * 0.1)
                max_lim = max_val * 1.1
                infinity_value = max_val * 1.15 if max_val > 0 else 1.0

                # Plot each finite homology dimension separately
                for i, (dim, points) in enumerate(sorted(group_finite.items())):
                    points = np.array(points)
                    color = colors[i % len(colors)]
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        s=40,
                        color=color,
                        alpha=0.5,
                        edgecolor=color,
                        label=custom_labels.get(dim, rf"$H_{{{dim}}}$")
                    )

                # --- Plot infinite persistence points ---
                for i, (dim, births) in enumerate(sorted(group_infinite.items())):
                    births = np.array(births)
                    color = colors[i % len(colors)]
                    # jitter so points don't overlap perfectly
                    y_vals = infinity_value + np.linspace(-0.02, 0.02, len(births))
                    # y_vals = np.full_like(births, infinity_value)
                    ax.scatter(
                        births,
                        y_vals,
                        s=40,
                        color=color,
                        alpha=0.5,
                        edgecolor=color,
                        # label=custom_labels.get(dim, rf"$H_{{{dim}}}$")
                    )

                # Diagonal line y = x
                ax.plot([min_lim, max_lim], [min_lim, max_lim], linestyle="-", color="black", linewidth=1)

                # Shade invalid region (below diagonal)
                ax.fill_between(
                    [min_lim, max_lim],
                    [min_lim, max_lim],
                    [min_lim, min_lim],
                    color="gray",
                    alpha=0.4
                )

                # Infinity line
                ax.axhline(infinity_value, linestyle="-", color="k", linewidth=0.5)
                current_ticks = list(ax.get_yticks())
                new_ticks = copy.deepcopy(current_ticks)
                new_ticks = new_ticks + [infinity_value] if infinity_value not in set(current_ticks) else new_ticks
                for t in current_ticks:
                    if (t*1.15) > infinity_value:
                        new_ticks.remove(t)
                new_labels = [str(int(t)) if t != infinity_value else r"$+\infty$" for t in new_ticks]
                ax.set_yticks(new_ticks)
                ax.set_yticklabels(new_labels)

                # Critical epsilon line
                if isinstance(critical_epsilon, float):
                    ax.axhline(critical_epsilon, linestyle="-", color="r", linewidth=1.2)
                    ax.text(5, critical_epsilon+0.5, r"$\epsilon_{critical}$", va='bottom', ha='left', color='r')

                ax.set_xlim(min_lim, max_lim)
                ax.set_ylim(min_lim, infinity_value * 1.1)

                ax.set_xlabel(r"Birth (Radius of $\epsilon$)")
                ax.set_ylabel(r"Death (Radius of $\epsilon$)")
                ax.set_title(r"Persistence Diagram ($\beta_0$ and $\beta_1$)")
                ax.legend(loc="lower right")
                # ax.grid(True, alpha=0.3)
                # return ax

            def plot_point_cloud(ax: plt.Axes):
                """Plot A: Pure node point cloud (Explicitly showing NO prior graph edges)"""
                if point_cloud is None:
                    return
                ax.scatter(point_cloud[:, 0], point_cloud[:, 1], color='k', s=16, zorder=5)
                ax.set_title("Point Cloud Data (Nodes)")
                ax.set_aspect('equal')
                # ax.grid(True, linestyle='--', alpha=0.5)

            # Get data
            ph_data = self._persistent_homology_data["persistent_homology"] # [(dim, (birth, death)), ...]
            point_cloud = self._persistent_homology_data["point_cloud_data"]
            if ph_data is None or point_cloud is None:
                return None

            critical_epsilon = None
            if self._results_df is not None:
                for _, row in self._results_df.iterrows():
                    if row["parameter"] == "Persistent Homology--Critical Epsilon":
                        critical_epsilon = row["value"]

            # Create plot figure
            plt_fig = plt.Figure(figsize=(8.5, 11), dpi=300)

            # Plot A: The Persistence Diagram tracking components and holes
            ax_1 = plt_fig.add_subplot(2, 1, 1)
            plot_persistence_diagram(ax=ax_1)  # gudhi.plot_persistence_diagram(persistence, axes=ax_1)

            # Plot B: Point Cloud Data
            ax_2 = plt_fig.add_subplot(2, 1, 2)
            plot_point_cloud(ax=ax_2)
            return plt_fig

        def plot_histograms():
            """
            Create plot figures of graph theory histograms selected by the user.

            :return: A list of Matplotlib figures.
            """
            self.update_status(ProgressData(percent=93, sender="GT", message=f"Generating histograms..."))

            opt_gte = graph_obj.configs
            plt_figs = []

            def plot_distribution_histogram(ax: Axes, title: str, distribution: list, x_label: str,
                                            plt_bins: np.ndarray|None = None, y_label: str = 'Counts'):
                """
                Create a histogram from a distribution dataset.

                :param ax: Plot axis.
                :param title: Title text.
                :param distribution: Dataset to be plotted.
                :param x_label: X-label title text.
                :param plt_bins: Bin dataset.
                :param y_label: Y-label title text.
                :return:
                """
                font_1 = {'fontsize': 9}
                if plt_bins is None:
                    plt_bins = np.linspace(min(distribution), max(distribution), 50)
                try:
                    std_val = str(round(stdev(distribution), 3))
                except StatisticsError:
                    std_val = "N/A"
                hist_title = title + std_val
                ax.set_title(hist_title, fontdict=font_1)
                ax.set(xlabel=x_label, ylabel=y_label)
                ax.hist(distribution, bins=plt_bins)

            # Degree and Closeness
            plt_fig = plt.Figure(figsize=(8.5, 11), dpi=300)
            if opt_gtc["display_degree_histogram"]["value"] == 1:
                deg_distribution = self._histogram_data["degree_distribution"]
                bins = np.arange(0.5, max(deg_distribution) + 1.5, 1)
                deg_title = r'Degree Distribution: $\sigma$='
                ax_1 = plt_fig.add_subplot(2, 1, 1)
                plot_distribution_histogram(ax_1, deg_title, deg_distribution, 'Degree', plt_bins=bins)

            if opt_gtc["display_closeness_centrality_histogram"]["value"] == 1:
                clo_distribution = self._histogram_data["closeness_distribution"]
                cc_title = r"Closeness Centrality: $\sigma$="
                ax_2 = plt_fig.add_subplot(2, 1, 2)
                plot_distribution_histogram(ax_2, cc_title, clo_distribution, 'Closeness value')
            plt_figs.append(plt_fig)

            # Betweenness, Clustering, Eigenvector and Ohms
            plt_fig = plt.Figure(figsize=(8.5, 11), dpi=300)
            if opt_gtc["display_betweenness_centrality_histogram"]["value"] == 1:
                bet_distribution = self._histogram_data["betweenness_distribution"]
                bc_title = r"Betweenness Centrality: $\sigma$="
                ax_1 = plt_fig.add_subplot(2, 2, 1)
                plot_distribution_histogram(ax_1, bc_title, bet_distribution, 'Betweenness value')

            if opt_gtc["compute_avg_clustering_coef"]["value"] == 1:
                cluster_coefs = self._histogram_data["clustering_coefficients"]
                clu_title = r"Clustering Coefficients: $\sigma$="
                ax_2 = plt_fig.add_subplot(2, 2, 2)
                plot_distribution_histogram(ax_2, clu_title, cluster_coefs, 'Clust. Coeff.')

            if opt_gtc["display_ohms_histogram"]["value"] == 1:
                ohm_distribution = self._histogram_data["ohms_distribution"]
                oh_title = r"Ohms Centrality: $\sigma$="
                ax_3 = plt_fig.add_subplot(2, 2, 3)
                plot_distribution_histogram(ax_3, oh_title, ohm_distribution, 'Ohms value')

            if opt_gtc["display_eigenvector_centrality_histogram"]["value"] == 1:
                eig_distribution = self._histogram_data["eigenvector_distribution"]
                ec_title = r"Eigenvector Centrality: $\sigma$="
                ax_4 = plt_fig.add_subplot(2, 2, 4)
                plot_distribution_histogram(ax_4, ec_title, eig_distribution, 'Eigenvector value')
            plt_figs.append(plt_fig)

            # weighted histograms
            if opt_gte["has_weights"]["value"] == 1:
                wt_type = graph_obj.get_weight_type()
                weight_type: str = str(wt_str) if (wt_str := FiberNetworkBuilder.get_weight_options().get(wt_type)) is not None else ""

                # degree, betweenness, closeness and eigenvector
                plt_fig = plt.Figure(figsize=(8.5, 11), dpi=300)
                if opt_gtc["display_degree_histogram"]["value"] == 1:
                    w_deg_distribution = self._histogram_data["weighted_degree_distribution"]
                    bins = np.arange(0.5, max(w_deg_distribution) + 1.5, 1)
                    w_deg_title = r"Weighted Degree: $\sigma$="
                    ax_1 = plt_fig.add_subplot(2, 2, 1)
                    plot_distribution_histogram(ax_1, w_deg_title, w_deg_distribution, 'Degree', plt_bins=bins)

                if opt_gtc["display_betweenness_centrality_histogram"]["value"] == 1:
                    w_bet_distribution = self._histogram_data["weighted_betweenness_distribution"]
                    w_bt_title = weight_type + r"-Weighted Betweenness: $\sigma$="
                    ax_2 = plt_fig.add_subplot(2, 2, 2)
                    plot_distribution_histogram(ax_2, w_bt_title, w_bet_distribution, 'Betweenness value')

                if opt_gtc["display_closeness_centrality_histogram"]["value"] == 1:
                    w_clo_distribution = self._histogram_data["weighted_closeness_distribution"]
                    w_clo_title = r"Length-Weighted Closeness: $\sigma$="
                    ax_3 = plt_fig.add_subplot(2, 2, 3)
                    plot_distribution_histogram(ax_3, w_clo_title, w_clo_distribution, 'Closeness value')

                if opt_gtc["display_eigenvector_centrality_histogram"]["value"] == 1:
                    w_eig_distribution = self._histogram_data["weighted_eigenvector_distribution"]
                    w_ec_title = weight_type + r"-Weighted Eigenvector Cent.: $\sigma$="
                    ax_4 = plt_fig.add_subplot(2, 2, 4)
                    plot_distribution_histogram(ax_4, w_ec_title, w_eig_distribution, 'Eigenvector value')
                plt_figs.append(plt_fig)

            return plt_figs

        def plot_heatmaps():
            """
            Create plot figures of graph theory heatmaps.

            :return: A list of Matplotlib figures.
            """
            self.update_status(ProgressData(percent=95, sender="GT", message=f"Generating heatmaps..."))

            sz = 30
            lc = 'black'
            plt_figs = []
            opt_gte = graph_obj.configs
            wt_type = graph_obj.get_weight_type()
            weight_type = FiberNetworkBuilder.get_weight_options().get(wt_type)

            def plot_distribution_heatmap(distribution: list, title: str, size: float, line_color: str):
                """
                Create a heatmap from a distribution.

                :param distribution: Dataset to be plotted.
                :param title: Title of the plot figure.
                :param size: Size of the scatter items.
                :param line_color: Color of the line items.
                :return: Histogram plot figure.
                """
                nx_graph = graph_obj.nx_giant_graph
                fig_grp = FiberNetworkBuilder.plot_graph_edges(img_3d, nx_graph, node_distribution_data=distribution,
                                                               plot_nodes=True, edge_color=line_color, node_marker_size=size)

                plt_fig_inner = fig_grp[0]
                plt_fig_inner.set_size_inches(8.5, 8.5)
                plt_fig_inner.set_dpi(400)
                plt_ax = plt_fig_inner.axes[0]
                plt_ax.set_title(title, fontdict={'fontsize': 9})
                plt_ax.set_position([0.05, 0.05, 0.75, 0.75])

                return plt_fig_inner

            if opt_gtc["display_degree_histogram"]["value"] == 1:
                deg_distribution = self._histogram_data["degree_distribution"]
                plt_fig = plot_distribution_heatmap(deg_distribution, 'Degree Heatmap', sz, lc)
                plt_figs.append(plt_fig)
            if (opt_gtc["display_degree_histogram"]["value"] == 1) and (opt_gte["has_weights"]["value"] == 1):
                w_deg_distribution = self._histogram_data["weighted_degree_distribution"]
                plt_title = 'Weighted Degree Heatmap'
                plt_fig = plot_distribution_heatmap(w_deg_distribution, plt_title, sz, lc)
                plt_figs.append(plt_fig)
            if opt_gtc["compute_avg_clustering_coef"]["value"] == 1:
                cluster_coefs = self._histogram_data["clustering_coefficients"]
                plt_title = 'Clustering Coefficient Heatmap'
                plt_fig = plot_distribution_heatmap(cluster_coefs, plt_title, sz, lc)
                plt_figs.append(plt_fig)
            if opt_gtc["display_betweenness_centrality_histogram"]["value"] == 1:
                bet_distribution = self._histogram_data["betweenness_distribution"]
                plt_title = 'Betweenness Centrality Heatmap'
                plt_fig = plot_distribution_heatmap(bet_distribution, plt_title, sz, lc)
                plt_figs.append(plt_fig)
            if (opt_gtc["display_betweenness_centrality_histogram"]["value"] == 1) and (
                    opt_gte["has_weights"]["value"] == 1):
                w_bet_distribution = self._histogram_data["weighted_betweenness_distribution"]
                plt_title = f'{weight_type}-Weighted Betweenness Centrality Heatmap'
                plt_fig = plot_distribution_heatmap(w_bet_distribution, plt_title, sz, lc)
                plt_figs.append(plt_fig)
            if opt_gtc["display_closeness_centrality_histogram"]["value"] == 1:
                clo_distribution = self._histogram_data["closeness_distribution"]
                plt_title = 'Closeness Centrality Heatmap'
                plt_fig = plot_distribution_heatmap(clo_distribution, plt_title, sz, lc)
                plt_figs.append(plt_fig)
            if (opt_gtc["display_closeness_centrality_histogram"]["value"] == 1) and (
                    opt_gte["has_weights"]["value"] == 1):
                w_clo_distribution = self._histogram_data["weighted_closeness_distribution"]
                plt_title = 'Length-Weighted Closeness Centrality Heatmap'
                plt_fig = plot_distribution_heatmap(w_clo_distribution, plt_title, sz, lc)
                plt_figs.append(plt_fig)
            if opt_gtc["display_eigenvector_centrality_histogram"]["value"] == 1:
                eig_distribution = self._histogram_data["eigenvector_distribution"]
                plt_title = 'Eigenvector Centrality Heatmap'
                plt_fig = plot_distribution_heatmap(eig_distribution, plt_title, sz, lc)
                plt_figs.append(plt_fig)
            if (opt_gtc["display_eigenvector_centrality_histogram"]["value"] == 1) and (
                    opt_gte["has_weights"]["value"] == 1):
                w_eig_distribution = self._histogram_data["weighted_eigenvector_distribution"]
                plt_title = f'{weight_type}-Weighted Eigenvector Centrality Heatmap'
                plt_fig = plot_distribution_heatmap(w_eig_distribution, plt_title, sz, lc)
                plt_figs.append(plt_fig)
            if opt_gtc["display_ohms_histogram"]["value"] == 1:
                ohm_distribution = self._histogram_data["ohms_distribution"]
                plt_title = 'Ohms Centrality Heatmap'
                plt_fig = plot_distribution_heatmap(ohm_distribution, plt_title, sz, lc)
                plt_figs.append(plt_fig)
            return plt_figs

        # 1. Compute graphs and plots for scaling behavior
        scaling_figs = plot_scaling_behavior()

        # 2. plotting the original, grayscale, and binary image, as well as the histogram of pixel grayscale values
        figs = plot_bin_images()
        for fig in figs:
            out_figs.append(fig)

        # 3a. plotting graph nodes
        fig = graph_obj.plot_graph_network(image_arr=img_3d, plot_nodes=True, a4_size=True)
        if fig is not None:
            out_figs.append(fig)

        # 4b. plotting graph edges
        fig = graph_obj.plot_graph_network(image_arr=img_3d, a4_size=True)
        if fig is not None:
            out_figs.append(fig)

        # 4a. displaying all the GT calculations in Table (on the entire page)
        fig, fig_wt = plot_gt_results()
        out_figs.append(fig)
        if fig_wt:
            out_figs.append(fig_wt)

        # 4b. display scaling GT results in a Table
        for fig in scaling_figs:
            out_figs.append(fig)

        # 5 display persistence homology diagram
        fig = plot_persistent_homology()
        if fig is not None:
            out_figs.append(fig)

        # 6a. displaying histograms
        figs = plot_histograms()
        for fig in figs:
            out_figs.append(fig)

        # 6b. displaying heatmaps
        if opt_gtc["display_heatmaps"]["value"] == 1:
            figs = plot_heatmaps()
            for fig in figs:
                out_figs.append(fig)

        # 7. displaying run information
        fig = plot_run_configs()
        out_figs.append(fig)
        return out_figs

    @staticmethod
    def write_to_pdf(sgt_obj, update_func=None) -> bool:
        """
        Write results to a PDF file.

        Args:
            sgt_obj: StructuralGT object with calculated GT parameters
            update_func: Callable for progress updates (e.g., update_func(percentage, message))

        Returns:
            True if the PDF file is written successfully, otherwise False
        """
        try:
            if update_func:
                update_func(ProgressData(percent=98, sender="GT", message="Writing PDF..."))

            filename, output_location = sgt_obj.ntwk_p.get_filenames()
            pdf_filename = filename + "_SGT_results.pdf"
            pdf_file = os.path.join(output_location, pdf_filename)

            if not sgt_obj.plot_figures:
                raise ValueError("No figures available to write to PDF.")

            with PdfPages(pdf_file) as pdf:  # type: ignore
                for fig in sgt_obj.plot_figures:
                    pdf.savefig(fig)

            if sgt_obj.results_df is not None:
                csv_filename = filename + "_SGT_unweighted.csv"
                csv_file = os.path.join(output_location, csv_filename)
                sgt_obj.results_df.to_csv(csv_file, index=False)

            if sgt_obj.weighted_results_df is not None:
                csv_filename = filename + "_SGT_weighted.csv"
                csv_file = os.path.join(output_location, csv_filename)
                sgt_obj.weighted_results_df.to_csv(csv_file, index=False)

            if sgt_obj.scaling_results:
                excel_filename = filename + "_SGT_scaling.xlsx"
                excel_file = os.path.join(output_location, excel_filename)
                with pd.ExcelWriter(str(excel_file), engine='xlsxwriter') as writer:
                    for tbl_title, tbl_df in sgt_obj.scaling_results.items():
                        # Clean sheet name: Excel allows max 31 chars, no : \ / ? * [ ]
                        safe_title = str(tbl_title)[:31].replace(":", "").replace("\\", "").replace("/", "") \
                            .replace("?", "").replace("*", "").replace("[", "").replace("]", "")
                        tbl_df.to_excel(writer, sheet_name=safe_title, index=False)
            return True
        except Exception as err:
            logging.exception("GT Computation Error: %s", err, extra={'user': 'SGT Logs'})
            if update_func:
                update_func(ProgressData(type="error", sender="GT", message="An error occurred while writing PDF."))
            return False

    @staticmethod
    def safe_run_analyzer(sgt_obj, update_func, save_to_pdf=False) -> tuple[bool, None] | tuple[bool, "GraphAnalyzer"]:
        """
        Safely compute GT descriptors without raising exceptions or crushing app.

        Args:
            sgt_obj: StructuralGT object with calculated GT parameters
            update_func: Callable for progress updates (e.g., update_func(msg_data))
            save_to_pdf: Save results to a PDF file
        """
        try:
            # Add Listeners
            sgt_obj.add_listener(update_func)

            # Run GT computations
            sgt_obj.run_analyzer()
            if sgt_obj.abort:
                raise AbortException("Process aborted")

            # Write GT results to PDF
            if save_to_pdf:
                GraphAnalyzer.write_to_pdf(sgt_obj, update_func)

            # Cleanup - remove listeners
            sgt_obj.remove_listener(update_func)
            return True, sgt_obj
        except AbortException:
            update_func(ProgressData(type="error", sender="GT", message="Task aborted by user of a fatal error occurred!"))
            sgt_obj.remove_listener(update_func)
            return False, None
        except Exception as err:
            update_func(ProgressData(type="error", sender="GT", message="Error encountered! Try again."))
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            # Clean up listeners before exiting
            sgt_obj.remove_listener(update_func)
            return False, None

    @staticmethod
    def safe_run_multi_analyzer(sgt_objs, update_func) -> None|dict:
        """
        Safely compute GT descriptors of multiple images without raising exceptions or crushing the app.

        Args:
            sgt_objs: List of StructuralGT objects with calculated GT parameters
            update_func: Callable for progress updates (e.g., update_func(msg_data))
        """
        try:
            i = 0
            keys_list = list(sgt_objs.keys())
            for key in keys_list:
                sgt_obj = sgt_objs[key]

                status_msg = f"Analyzing Image: {(i + 1)} / {len(sgt_objs)}"
                update_func(ProgressData(type="info", sender="GT", message=status_msg))

                start = time.time()
                success, new_sgt = GraphAnalyzer.safe_run_analyzer(sgt_obj, update_func)
                # TerminalApp.is_aborted(sgt_obj)
                if success:
                    GraphAnalyzer.write_to_pdf(new_sgt, update_func)
                end = time.time()

                i += 1
                output = status_msg + "\n" + f"Run-time: {str(end - start)}  seconds\n"
                output += "Results generated for: " + sgt_obj.ntwk_p.img_path + "\n"
                # graph_obj = sgt_obj.ntwk_p.graph_obj
                # output += "Node Count: " + str(graph_obj.nx_giant_graph.number_of_nodes()) + "\n"
                # output += "Edge Count: " + str(graph_obj.nx_giant_graph.number_of_edges()) + "\n"
                # filename, out_dir = sgt_obj.ntwk_p.get_filenames()
                # out_file = os.path.join(out_dir, filename + '-v2_results.txt')
                # write_txt_file(output, out_file)
                logging.info(output, extra={'user': 'SGT Logs'})
            return sgt_objs
        except AbortException:
            update_func(ProgressData(type="error", sender="GT", message="Task aborted by user of a fatal error occurred!"))
            return None
        except Exception as err:
            update_func(ProgressData(type="error", sender="GT", message="Error encountered! Try again."))
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            return None
