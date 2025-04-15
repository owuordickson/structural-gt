# SPDX-License-Identifier: GNU GPL v3

"""
Builds a graph network from nano-scale microscopy images.
"""

import os
import io

import igraph
from PIL import Image
import sknw
import logging
import itertools
import numpy as np
import networkx as nx
from PIL.ImageFile import ImageFile
from cv2.typing import MatLike
import matplotlib.pyplot as plt

from .progress_update import ProgressUpdate
from .graph_skeleton import GraphSkeleton
from .sgt_utils import write_csv_file, write_gsd_file
from ..configs.config_loader import load_gte_configs

# WE ARE USING CPU BECAUSE CuPy generates some errors - yet to be resolved.
COMPUTING_DEVICE = "CPU"
try:
    import sys

    logger = logging.getLogger("SGT App")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
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


class GraphExtractor(ProgressUpdate):
    """
    A class for builds a graph network from microscopy images and stores is as a NetworkX object.

    """

    def __init__(self):
        """
        A class for builds a graph network from microscopy images and stores is as a NetworkX object.

        """
        super(GraphExtractor, self).__init__()
        self.configs: dict = load_gte_configs()  # graph extraction parameters and options.
        self.props: list = []
        self.img_ntwk: ImageFile | None = None
        self.nx_graph = None
        self.ig_graph = None
        self.skel_obj: GraphSkeleton | None = None
        self.nx_components = []

    def fit_graph(self, image_bin: MatLike = None, image_2d: MatLike = None, px_width_sz: float = 1.0, rho_val: float = 1.0):
        """
        Execute a function that builds a NetworkX graph from the binary image.

        :param image_bin: a binary image for building Graph Skeleton for the NetworkX graph.
        :param image_2d: the raw 2D image for creating a visual graph plot image.
        :param px_width_sz: width of a pixel in nano-meters.
        :param rho_val: resistivity coefficient/value of the material.
        :return:
        """

        if self.abort:
            self.update_status([-1, "Task aborted by due to an error. If problem with graph: change/apply different "
                                    "image/binary filters and graph options. OR change brightness/contrast"])
            return

        self.update_status([50, "Making graph skeleton..."])
        success = self.extract_graph(image_bin=image_bin, px_size=px_width_sz, rho_val=rho_val)
        if not success:
            self.update_status([-1, "Problem encountered, provide GT parameters"])
            self.abort = True
            return

        self.update_status([75, "Verifying graph network..."])
        if self.nx_graph.number_of_nodes() <= 0:
            self.update_status([-1, "Problem generating graph (change image/binary filters)"])
            self.abort = True
            return

        self.update_status([77, "Retrieving graph properties..."])
        self.props = self.get_graph_props()

        self.update_status([90, "Drawing graph network..."])
        graph_plt = self.plot_2d_graph_network(image_2d=image_2d)
        if graph_plt is not None:
            self.img_ntwk = GraphExtractor.plot_to_img(graph_plt)

    def reset_graph(self):
        """
        Erase the existing data stored in the object.
        :return:
        """
        self.nx_graph, self.ig_graph, self.img_ntwk = None, None, None

    def extract_graph(self, image_bin: MatLike = None, px_size: float = 1.0, rho_val: float = 1.0):
        """
        Build a skeleton from image and use the skeleton to build a NetworkX graph.

        :param image_bin: binary image from which skeleton will be built and graph drawn.
        :param px_size: width of a pixel in nano-meters.
        :param rho_val: resistivity coefficient/value of the material.
        :return:
        """

        if image_bin is None:
            return False

        opt_gte = self.configs
        if opt_gte is None:
            return False

        graph_skel = GraphSkeleton(image_bin, opt_gte)
        self.skel_obj = graph_skel
        img_skel = graph_skel.skeleton.astype(int)  # DOES NOT PLOT - node_plot (BUG IN CODE), so we pick first image in stack
        selected_slice = 0  # Select first slice in 3D skeleton of shape (depth, w, h)

        self.update_status([60, "Creating graph network..."])
        nx_graph = sknw.build_sknw(img_skel[selected_slice])
        for (s, e) in nx_graph.edges():
            # 'sknw' library stores length of edge and calls it weight, we reverse this
            # we create a new attribute 'length', later delete/modify 'weight'
            nx_graph[s][e]['length'] = nx_graph[s][e]['weight']
            #    if nx_graph[s][e]['weight'] == 0:  # TO BE DELETED later
            #        nx_graph[s][e]['length'] = 2
            ge = nx_graph[s][e]['pts']

            if opt_gte["has_weights"]["value"] == 1:
                # We modify 'weight'
                wt_type = self.get_weight_type()
                weight_options = GraphExtractor.get_weight_options()
                pix_width, pix_angle, wt = graph_skel.assign_weights(ge, wt_type, weight_options=weight_options,
                                                                         pixel_dim=px_size, rho_dim=rho_val)
            else:
                pix_width, pix_angle, wt = graph_skel.assign_weights(ge, None)
                # delete 'weight'
                # del nx_graph[s][e]['weight']
            nx_graph[s][e]['width'] = pix_width
            nx_graph[s][e]['angle'] = pix_angle
            nx_graph[s][e]['weight'] = wt
            # print(f"{nx_graph[s][e]}\n")
        self.nx_graph = nx_graph
        self.ig_graph = igraph.Graph.from_networkx(nx_graph)

        # Removing all instances of edges were the start and end are the same, or "self loops"
        if opt_gte["remove_self_loops"]["value"]:
            for (s, e) in self.nx_graph.edges():
                if s == e:
                    self.nx_graph.remove_edge(s, e)
        return True

    def plot_2d_graph_network(self, image_2d: MatLike = None, a4_size: bool = False, blank: bool = False):
        """
        Creates a plot figure of the graph network. It draws all the edges and nodes of the graph.

        :param image_2d: 2D image to be used to draw the network.
        :param a4_size: decision if to create an A4 size plot figure.
        :param blank: do not add image in the background, have a white background.

        :return:
        """

        if image_2d is None:
            return None

        nx_graph = self.nx_graph
        nx_components = self.nx_components

        if blank:
            w, h = image_2d.shape[:2]
            my_dpi = 96
            fig = plt.Figure(figsize=(h / my_dpi, w / my_dpi), dpi=my_dpi)
            ax = fig.add_axes((0, 0, 1, 1))  # span the whole figure
            ax.axis('off')
            ax.imshow(image_2d, cmap='gray', alpha=0)
            for (s, e) in nx_graph.edges():
                ge = nx_graph[s][e]['pts']
                ax.plot(ge[:, 1], ge[:, 0], 'black')
            nodes = nx_graph.nodes()
            gn = xp.array([nodes[i]['o'] for i in nodes])
            ax.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)

            img = xp.array(GraphExtractor.plot_to_img(fig))
            if len(img.shape) == 3:
                img = xp.mean(img[:, :, :2], 2)  # Convert the image to grayscale (or 2D)
            return img

        if len(nx_components) > 0:
            if a4_size:
                fig = plt.Figure(figsize=(8.5, 11), dpi=400)
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title("Graph Sub-networks")
            else:
                fig = plt.Figure()
                ax = fig.add_axes((0, 0, 1, 1))  # span the whole figure
            ax.set_axis_off()
            ax.imshow(image_2d, cmap='gray')
            color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            color_cycle = itertools.cycle(color_list)
            for component in nx_components:
                sg = nx_graph.subgraph(component)
                color = next(color_cycle)
                for (s, e) in sg.edges():
                    ge = sg[s][e]['pts']
                    ax.plot(ge[:, 1], ge[:, 0], color)
        else:
            if a4_size:
                return None
            else:
                fig = plt.Figure()
                ax = fig.add_axes((0, 0, 1, 1))  # span the whole figure
            ax.set_axis_off()
            GraphExtractor.superimpose_graph_to_img(ax, image_2d, nx_graph)

        return fig

    def plot_2d_skeletal_images(self, image_2d: MatLike = None):
        """
        Create plot figures of skeletal image and graph network image.

        :param image_2d: raw 2D image to be super-imposed with graph.

        :return:
        """

        if image_2d is None:
            return None

        opt_gte = self.configs
        nx_graph = self.nx_graph
        g_skel = self.skel_obj

        fig = plt.Figure(figsize=(8.5, 11), dpi=400)
        ax_1 = fig.add_subplot(2, 1, 1)
        ax_2 = fig.add_subplot(2, 1, 2)

        ax_1.set_title("Skeletal Image")
        ax_1.set_axis_off()
        ax_1.imshow(g_skel.skel_int, cmap='gray')
        ax_1.scatter(g_skel.bp_coord_x, g_skel.bp_coord_y, s=0.25, c='b')
        ax_1.scatter(g_skel.ep_coord_x, g_skel.ep_coord_y, s=0.25, c='r')

        ax_2.set_title("Final Graph")
        ax_2.set_axis_off()
        ax_2 = GraphExtractor.superimpose_graph_to_img(ax_2, image_2d, nx_graph)

        nodes = nx_graph.nodes()
        gn = xp.array([nodes[i]['o'] for i in nodes])
        if opt_gte["display_node_id"]["value"] == 1:
            i = 0
            for x, y in zip(gn[:, 1], gn[:, 0]):
                ax_2.annotate(str(i), (x, y), fontsize=5)
                i += 1
            ax_2.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)
        else:
            ax_2.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)
        return fig

    def get_config_info(self):
        """
        Get the user selected parameters and options information.
        :return:
        """

        opt_gte = self.configs

        run_info = "***Graph Extraction Configurations***\n"
        if opt_gte["has_weights"]["value"] == 1:
            wt_type = self.get_weight_type()
            run_info += f"Weight Type: {GraphExtractor.get_weight_options().get(wt_type)} || "
        if opt_gte["merge_nearby_nodes"]["value"]:
            run_info += "Merge Nodes || "
        if opt_gte["prune_dangling_edges"]["value"]:
            run_info += "Prune Dangling Edges || "
        run_info = run_info[:-3] + '' if run_info.endswith('|| ') else run_info
        run_info += "\n"
        if opt_gte["remove_disconnected_segments"]["value"]:
            run_info += f"Remove Objects of Size = {opt_gte["remove_disconnected_segments"]["items"][0]["value"]} || "
        if opt_gte["remove_self_loops"]["value"]:
            run_info += "Remove Self Loops || "
        run_info = run_info[:-3] + '' if run_info.endswith('|| ') else run_info

        return run_info

    def get_graph_props(self):
        """
        A method that retrieves graph properties and stores them in a list-array.

        Returns: list of graph properties
        """

        graph = self.nx_graph

        # 1. Identify the subcomponents (graph segments) that make up the entire NetworkX graph.
        self.update_status([78, "Identifying graph subcomponents..."])
        connected_components = list(nx.connected_components(graph))
        if not connected_components:  # In case the graph is empty
            connected_components = []
        sub_graphs = [graph.subgraph(c).copy() for c in connected_components]
        giant_graph = max(sub_graphs, key=lambda g: g.number_of_nodes())
        num_graphs = len(sub_graphs)
        self.nx_components = connected_components
        connect_ratio = giant_graph.number_of_nodes() / graph.number_of_nodes()

        # 2. Populate graph properties
        self.update_status([80, "Storing graph properties..."])
        props = [
            ["Weight Type", str(GraphExtractor.get_weight_options().get(self.get_weight_type()))],
            ["Edge Count", str(graph.number_of_edges())],
            ["Node Count", str(graph.number_of_nodes())],
            ["Graph Count", str(len(self.nx_components))],
            ["Sub-graph Count", str(num_graphs)],
            ["Giant-to-Entire graph ratio", f"{round((connect_ratio * 100), 3)}%"]]
        return props

    def get_weight_type(self):
        wt_type = None  # Default weight
        if self.configs["has_weights"]["value"] == 0:
            return wt_type

        for i in range(len(self.configs["has_weights"]["items"])):
            if self.configs["has_weights"]["items"][i]["value"]:
                wt_type = self.configs["has_weights"]["items"][i]["id"]
        return wt_type

    def save_graph_to_file(self, filename: str, out_dir: str):
        """
        Save graph data into files.

        :param filename: the filename to save the data to.
        :param out_dir: the directory to save the data to.
        :return:
        """

        nx_graph = self.nx_graph.copy()
        opt_gte = self.configs

        g_filename = filename + "_graph.gexf"
        el_filename = filename + "_EL.csv"
        adj_filename = filename + "_adj.csv"
        gsd_filename = filename + "_skel.gsd"
        gexf_file = os.path.join(out_dir, g_filename)
        csv_file = os.path.join(out_dir, el_filename)
        adj_file = os.path.join(out_dir, adj_filename)

        if opt_gte["export_adj_mat"]["value"] == 1:
            adj_mat = nx.adjacency_matrix(self.nx_graph).todense()
            xp.savetxt(str(adj_file), adj_mat, delimiter=",")

        if opt_gte["export_edge_list"]["value"] == 1:
            if opt_gte["has_weights"]["value"] == 1:
                fields = ['Source', 'Target', 'Weight', 'Length']
                el = nx.generate_edgelist(nx_graph, delimiter=',', data=True)
                write_csv_file(csv_file, fields, el)
            else:
                fields = ['Source', 'Target']
                el = nx.generate_edgelist(nx_graph, delimiter=',', data=False)
                write_csv_file(csv_file, fields, el)

        if opt_gte["export_as_gexf"]["value"] == 1:
            # deleting extraneous info and then exporting the final skeleton
            for (x) in nx_graph.nodes():
                del nx_graph.nodes[x]['pts']
                del nx_graph.nodes[x]['o']
            for (s, e) in nx_graph.edges():
                del nx_graph[s][e]['pts']
            nx.write_gexf(nx_graph, gexf_file)

        if opt_gte["export_as_gsd"]["value"] == 1:
            gsd_file = os.path.join(out_dir, gsd_filename)
            if self.skel_obj.skeleton is not None:
                write_gsd_file(gsd_file, self.skel_obj.skeleton)

    @staticmethod
    def plot_to_img(fig: plt.Figure):
        """
        Convert a Matplotlib figure to a PIL Image and return it

        :param fig: Matplotlib figure.
        """
        if fig:
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
            return img

    @staticmethod
    def get_weight_options():
        """
        Returns the weight options for building the graph edges.

        :return:
        """
        weight_options = {
            'DIA': 'Diameter',
            'AREA': 'Area',  # surface area of edge
            'LEN': 'Length',
            'ANGLE': 'Angle',
            'INV_LEN': 'InverseLength',
            'VAR_CON': 'Conductance',  # with variable width
            'FIX_CON': 'FixedWidthConductance',
            'RES': 'Resistance',
            # '': ''
        }
        return weight_options

    @staticmethod
    def superimpose_graph_to_img(axis, image: MatLike, nx_graph: nx.Graph):
        """
        Plot graph edges on top of the image.
        :param axis: matplotlib axis
        :param image: image to be superimposed with graph edges
        :param nx_graph: a NetworkX graph
        :return:
        """
        # axis.set_axis_off()
        axis.imshow(image, cmap='gray')
        for (s, e) in nx_graph.edges():
            ge = nx_graph[s][e]['pts']
            axis.plot(ge[:, 1], ge[:, 0], 'red')
        return axis
