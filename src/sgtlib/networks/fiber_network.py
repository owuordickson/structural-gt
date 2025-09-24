# SPDX-License-Identifier: GNU GPL v3

"""
Builds a graph network from nanoscale microscopy images.
"""

import os
import numbers
import itertools
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
from cv2.typing import MatLike

from .sknw_mod import build_sknw#, build_graph
from ..networks.graph_skeleton import GraphSkeleton
from ..utils.config_loader import load_gte_configs
from ..utils.sgt_utils import write_gsd_file, gsd_to_skeleton, csv_to_graph, ProgressUpdate, ProgressData


class FiberNetworkBuilder(ProgressUpdate):
    """
    A class for builds a graph network from microscopy images and stores is as a NetworkX object.

    """

    def __init__(self, cfg_file=""):
        """
        A class for builds a graph network from microscopy images and stores is as a NetworkX object.

        Args:
            cfg_file (str): configuration file path

        """
        super(FiberNetworkBuilder, self).__init__()
        self._configs: dict = load_gte_configs(cfg_file)  # graph extraction parameters and options.
        self._props: list = []
        self._img_ntwk: MatLike | None = None
        self._score_rating: float = -1.0
        self._nx_giant_graph: nx.Graph | None = None
        self._nx_graph: nx.Graph | None = None
        self._ig_graph: None | ig.Graph = None
        self._gsd_file: str | None = None
        self._skel_obj: GraphSkeleton | None = None

    @property
    def configs(self) -> dict:
        """Returns the graph extraction configuration parameters and options."""
        return self._configs

    @configs.setter
    def configs(self, configs: dict):
        """Sets the graph extraction configuration parameters and options."""
        self._configs = configs

    @property
    def props(self) -> list:
        """Returns the computed graph properties as a list-array."""
        return self._props

    @property
    def img_ntwk(self) -> MatLike | None:
        """Returns the processed image with the graph drawn on it."""
        return self._img_ntwk

    @img_ntwk.setter
    def img_ntwk(self, img_ntwk: MatLike | None):
        """Sets the processed image with the graph drawn on it."""
        self._img_ntwk = img_ntwk

    @property
    def nx_giant_graph(self) -> nx.Graph | None:
        """Returns the giant graph of the NetworkX object."""
        return self._nx_giant_graph

    @nx_giant_graph.setter
    def nx_giant_graph(self, nx_giant_graph: nx.Graph | None):
        """Sets the giant graph of the NetworkX object."""
        self._nx_giant_graph = nx_giant_graph

    @property
    def nx_graph(self) -> nx.Graph | None:
        """Returns the NetworkX graph object."""
        return self._nx_graph

    @nx_graph.setter
    def nx_graph(self, nx_graph: nx.Graph | None):
        """Sets the NetworkX graph object."""
        self._nx_graph = nx_graph

    @property
    def ig_graph(self) -> None | ig.Graph:
        """Returns the iGraph graph object."""
        return self._ig_graph

    @property
    def gsd_file(self) -> str | None:
        """Returns the filename of the graph skeleton saved in GSD format."""
        return self._gsd_file

    @property
    def skel_obj(self):
        return self._skel_obj

    @property
    def score_rating(self) -> float:
        """Returns the score rating of the graph."""
        return self._score_rating

    @score_rating.setter
    def score_rating(self, score: float):
        """Sets the score rating of the graph."""
        if 0 <= score <= 100.0:
            self._score_rating = score

    def fit_graph(self, save_dir: str, input_data: MatLike | str = None, is_img_2d: bool = True, px_width_sz: float = 1.0, rho_val: float = 1.0, file_name: str = "img") -> None:
        """
        Execute functions that build a NetworkX graph from the binary image.

        :param save_dir: Directory to save the graph to.
        :param input_data: A binary image for building Graph Skeleton for the NetworkX graph OR a file path for loading the NetworkX graph from a file.
        :param is_img_2d: Whether the image is 2D or 3D otherwise.
        :param px_width_sz: Width of a pixel in nanometers.
        :param rho_val: Resistivity coefficient/value of the material.
        :param file_name: Filename of the binary image.
        :return:
        """

        if self.abort:
            msg_data = ProgressData(
                type="error",
                sender="GT",
                message="Task aborted by due to an error. "
                        "If problem with graph: change/apply different image/binary filters and graph options. "
                        "OR change brightness/contrast")
            self.update_status(msg_data)
            return

        if type(input_data) is str:
            self.update_status(ProgressData(percent=50, sender="GT", message=f"Loading graph network from file..."))
            nx_graph = self.create_graph_from_file(input_data)
        elif type(input_data) is np.ndarray:
            self.update_status(ProgressData(percent=50, sender="GT", message=f"Extracting the graph network..."))
            nx_graph = self.extract_graph(image_bin=input_data, is_img_2d=is_img_2d, px_size=px_width_sz, rho_val=rho_val)
        else:
            msg_data = ProgressData(
                type="error",
                sender="GT",
                message="Invalid input for building a graph network. Either provide a graph file path or a binary image as an OpenCV matrix.")
            self.update_status(msg_data)
            self.abort = True
            return

        self.update_status(ProgressData(percent=70, sender="GT", message=f"Verifying graph network..."))
        success = self.verify_graph(nx_graph)
        if not success:
            msg_data = ProgressData(
                type="error",
                sender="GT",
                message="Problem encountered, change image/binary filters and graph options. OR change brightness/contrast")
            self.update_status(msg_data)
            self.abort = True
            return

        self.update_status(ProgressData(percent=77, sender="GT", message=f"Retrieving graph properties..."))
        self._props = self.get_graph_props()

        self.update_status(ProgressData(percent=90, sender="GT", message=f"Saving graph network..."))
        # Save graph to GSD/HOOMD - For OVITO rendering
        self._configs["export_as_gsd"]["value"] = 1
        self.save_graph_to_file(file_name, save_dir)

    def reset_graph(self) -> None:
        """
        Erase the existing data stored in the object.
        :return:
        """
        self.nx_graph, self._ig_graph, self._img_ntwk = None, None, None

    def verify_graph(self, nx_graph) -> bool:
        """
        Verify if the NetworkX graph is valid. If it is valid, save in object members.

        :param nx_graph: The NetworkX graph to verify.
        :return: True if the graph is valid, False otherwise.
        """
        if nx_graph is None:
            return False

        if nx_graph.number_of_edges() <= 0 or nx_graph.number_of_nodes() <= 0:
            return False

        # Save NetworkX graph
        self.nx_graph = nx_graph
        # Save iGraph graph
        self._ig_graph = ig.Graph.from_networkx(nx_graph)
        # Save giant NetworkX graph
        connected_components = list(nx.connected_components(nx_graph))
        if not connected_components:  # In case the graph is empty
            connected_components = []
        sub_graphs = [nx_graph.subgraph(c).copy() for c in connected_components]
        if sub_graphs:
            giant_graph = max(sub_graphs, key=lambda g: g.number_of_nodes())
        else:
            giant_graph = nx_graph
        self._nx_giant_graph = giant_graph
        return True

    def extract_graph(self, image_bin: MatLike = None, is_img_2d: bool = True, px_size: float = 1.0, rho_val: float = 1.0) -> nx.Graph | None:
        """
        Build a skeleton from the image and use the skeleton to build a NetworkX graph.

        :param image_bin: Binary image from which the skeleton will be built and graph drawn.
        :param is_img_2d: Whether the image is 2D or 3D otherwise.
        :param px_size: Width of a pixel in nanometers.
        :param rho_val: Resistivity coefficient/value of the material.
        :return:
        """
        try:
            if image_bin is None:
                return None

            opt_gte = self._configs
            if opt_gte is None:
                return None

            self.update_status(ProgressData(percent=51, sender="GT", message=f"Building skeleton from binary image..."))
            graph_skel = GraphSkeleton(image_bin, opt_gte, is_2d=is_img_2d, progress_func=self.update_status)
            self._skel_obj = graph_skel
            img_skel = graph_skel.skeleton

            self.update_status(ProgressData(percent=60, sender="GT", message=f"Creating graph network..."))
            # nx_graph = sknw.build_sknw(img_skel)
            nx_graph = build_sknw(img_skel)

            if opt_gte["remove_self_loops"]["value"]:
                self.update_status(ProgressData(percent=64, sender="GT", message=f"Removing self loops from graph network..."))

            self.update_status(ProgressData(percent=66, sender="GT", message=f"Assigning weights to graph network..."))
            for (s, e) in list(nx_graph.edges()):
                if opt_gte["remove_self_loops"]["value"]:
                    # Removing all instances of edges where the start and end are the same, or "self-loops"
                    if s == e:
                        nx_graph.remove_edge(s, e)
                        continue

                # 'sknw' library stores the length of edge as 'weight', we create an attribute 'length', and update 'weight'
                nx_graph[s][e]['length'] = nx_graph[s][e]['weight']
                ge = nx_graph[s][e]['pts']

                if opt_gte["has_weights"]["value"] == 1:
                    # We update 'weight'
                    wt_type = self.get_weight_type()
                    weight_options = FiberNetworkBuilder.get_weight_options()
                    pix_width, pix_angle, wt = graph_skel.assign_weights(ge, wt_type, weight_options=weight_options,
                                                                             pixel_dim=px_size, rho_dim=rho_val)
                    nx_graph[s][e]['width'] = pix_width
                    nx_graph[s][e]['angle'] = pix_angle
                    nx_graph[s][e]['weight'] = wt
                else:
                    pix_width, pix_angle, wt = graph_skel.assign_weights(ge, None)
                    nx_graph[s][e]['width'] = pix_width
                    nx_graph[s][e]['angle'] = pix_angle
                    del nx_graph[s][e]['weight']            # delete 'weight'
                # print(f"{nx_graph[s][e]}\n")

            return nx_graph
        except Exception as e:
            msg_data = ProgressData(
                type="error",
                message=f"Problem encountered while extracting graph from binary image: {e}")
            self.update_status(msg_data)
            return None

    def create_graph_from_file(self, file_path: str) -> nx.Graph | None:
        """
        Load a NetworkX graph from a file that may contain:
          - Edge list (2 columns)
          - Adjacency matrix (square matrix)
          - XYZ positions (3 columns: x, y, z, edges inferred by distance threshold)

        :param file_path: Path to the graph file
        :return: True if the graph is read, False otherwise
        """
        try:
            self.update_status(ProgressData(percent=60, sender="GT", message=f"Reading graph network..."))
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".gsd":
                skel = gsd_to_skeleton(file_path)
                self._skel_obj = GraphSkeleton(np.array([None]))
                self._skel_obj._skeleton = skel
                # self._skel_obj._skeleton_3d = np.asarray([skel])
                nx_graph = build_sknw(skel)
                for (s, e) in list(nx_graph.edges()):
                    nx_graph[s][e]['length'] = nx_graph[s][e]['weight']
                    nx_graph[s][e]['width'] = 1.0
            elif ext == ".csv":
                nx_graph = csv_to_graph(file_path)
                # self._skel_obj = GraphSkeleton(np.array([None]))
                # self._skel_obj._skeleton = skel
                ## self._skel_obj._skeleton_3d = np.asarray([skel])
                # nx_graph = build_sknw(skel)
            else:
                msg_data = ProgressData(
                    type="error",
                    message=f"Unsupported file extension: {ext}. If CSV, comma as delimiter.")
                self.update_status(msg_data)
                nx_graph = None
            return nx_graph
        except Exception as e:
            msg_data = ProgressData(
                type="error",
                sender="GT",
                message=f"Problem encountered while loading graph from file: {e}")
            self.update_status(msg_data)
            return None

    def plot_graph_network(self, image_arr: MatLike, giant_only: bool = False, plot_nodes: bool = False, a4_size: bool = False) -> None | plt.Figure:
        """
        Creates a plot figure of the graph network. It draws all the edges and nodes of the graph.

        :param image_arr: Slides of 2D images to be used to draw the network.
        :param giant_only: If True, only the giant graph is identified and drawn.
        :param plot_nodes: Make the graph's node plot figure.
        :param a4_size: Decision if to create an A4 size plot figure.

        :return:
        """

        if self.nx_graph is None:
            return None

        # Fetch the graph and config options
        if giant_only:
            nx_graph = self._nx_giant_graph
        else:
            nx_graph = self.nx_graph
        show_node_id = (self._configs["display_node_id"]["value"] == 1)
        add_width_thickness = (self._configs["add_width_thickness"]["value"] == 1)

        # Create the plot figure(s)
        fig_grp = FiberNetworkBuilder.plot_graph_edges(image_arr, nx_graph, plot_nodes=plot_nodes, show_node_id=show_node_id, add_width_thickness=add_width_thickness)
        fig = fig_grp[0]
        if a4_size:
            plt_title = "Graph Node Plot" if plot_nodes else "Graph Edge Plot"
            fig.set_size_inches(8.5, 11)
            fig.set_dpi(400)
            ax = fig.axes[0]
            ax.set_title(plt_title)
            # This moves the Axes to start: 5% from the left, 5% from the bottom,
            # and have a width and height: 80% of the figure.
            # [left, bottom, width, height]
            ax.set_position([0.05, 0.05, 0.9, 0.9])
        return fig

    def get_config_info(self) -> str:
        """
        Get the user selected parameters and options information.
        :return:
        """

        opt_gte = self._configs

        run_info = "***Graph Extraction Configurations***\n"
        if opt_gte["has_weights"]["value"] == 1:
            wt_type = self.get_weight_type()
            run_info += f"Weight Type: {FiberNetworkBuilder.get_weight_options().get(wt_type)} || "
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

    def get_graph_props(self) -> list:
        """
        A method that retrieves graph properties and stores them in a list-array.

        Returns: list of graph properties
        """

        # 1. Identify the subcomponents (graph segments) that make up the entire NetworkX graph.
        self.update_status(ProgressData(percent=78, sender="GT", message=f"Identifying graph subcomponents..."))
        graph = self.nx_graph.copy()
        connected_components = list(nx.connected_components(graph))
        if not connected_components:  # In case the graph is empty
            connected_components = []
        sub_graphs = [graph.subgraph(c).copy() for c in connected_components]
        num_graphs = len(sub_graphs)
        connect_ratio = self._nx_giant_graph.number_of_nodes() / graph.number_of_nodes()

        # 2. Populate graph properties
        self.update_status(ProgressData(percent=80, sender="GT", message=f"Storing graph properties..."))
        props = [
            ["Weight Type", str(FiberNetworkBuilder.get_weight_options().get(self.get_weight_type()))],
            ["Edge Count", str(graph.number_of_edges())],
            ["Node Count", str(graph.number_of_nodes())],
            ["Graph Count", str(len(connected_components))],
            ["Sub-graph Count", str(num_graphs)],
            ["Giant graph ratio", f"{round((connect_ratio * 100), 3)}%"],
            ["Accuracy Score", f"{self._score_rating}%" if self._score_rating > 0 else "N/A"]
        ]
        return props

    def get_weight_type(self) -> str | None:
        wt_type = None  # Default weight
        if self._configs["has_weights"]["value"] == 0:
            return wt_type

        for i in range(len(self._configs["has_weights"]["items"])):
            if self._configs["has_weights"]["items"][i]["value"]:
                wt_type = self._configs["has_weights"]["items"][i]["id"]
        return wt_type

    def save_graph_to_file(self, filename: str, out_dir: str) -> None:
        """
        Save graph data into files.

        :param filename: The filename to save the data to.
        :param out_dir: The directory to save the data to.
        :return:
        """

        nx_graph = self.nx_graph.copy()
        opt_gte = self._configs

        g_filename = filename + "_graph.gexf"
        edges_filename = filename + "_EdgeList.csv"
        nodes_filename = filename + "_NodePositions.csv"
        adj_filename = filename + "_AdjMat.csv"
        gsd_filename = filename + "_skel.gsd"
        gexf_file = os.path.join(out_dir, g_filename)
        edges_file = os.path.join(out_dir, edges_filename)
        nodes_file = os.path.join(out_dir, nodes_filename)
        adj_file = os.path.join(out_dir, adj_filename)

        if opt_gte["export_adj_mat"]["value"] == 1:
            adj_mat = nx.adjacency_matrix(self.nx_graph).todense()
            np.savetxt(str(adj_file), adj_mat, delimiter=",")

        if opt_gte["export_edge_list"]["value"] == 1:
            if opt_gte["has_weights"]["value"] == 1:
                cols = ['Source', 'Target', 'Weight', 'Length', 'Width', 'Angle']
                lst_edges = []
                for (s, e) in list(nx_graph.edges()):
                    weight = (nx_graph[s][e]['weight'])
                    length = nx_graph[s][e]['length']
                    width = nx_graph[s][e]['width']
                    angle = nx_graph[s][e]['angle']
                    el = [s, e, weight, length, width, angle]
                    lst_edges.append(el)
                df_edges = pd.DataFrame(lst_edges, columns=cols)
                df_edges.to_csv(edges_file, index=False)
            else:
                cols = ['Source', 'Target']
                lst_edges = []
                for (s, e) in list(nx_graph.edges()):
                    el = [s, e]
                    lst_edges.append(el)
                df_edges = pd.DataFrame(lst_edges, columns=cols)
                df_edges.to_csv(edges_file, index=False)

        if opt_gte["export_node_positions"]["value"] == 1:
            node_list = list(nx_graph.nodes())
            node_pos = np.array([nx_graph.nodes[i]['o'] for i in node_list])
            df_node_pos = pd.DataFrame(node_pos, columns=['x', 'y'])
            df_node_pos.to_csv(nodes_file, index=False)

        if opt_gte["export_as_gexf"]["value"] == 1:
            # deleting extraneous info and then exporting the final skeleton
            for (x) in nx_graph.nodes():
                del nx_graph.nodes[x]['pts']
                del nx_graph.nodes[x]['o']
            for (s, e) in nx_graph.edges():
                del nx_graph[s][e]['pts']
            nx.write_gexf(nx_graph, gexf_file)

        if opt_gte["export_as_gsd"]["value"] == 1:
            self._gsd_file = os.path.join(out_dir, gsd_filename)
            if self._skel_obj is not None:
                if self._skel_obj.skeleton_3d is not None:
                    write_gsd_file(self._gsd_file, self._skel_obj.skeleton_3d)

    @staticmethod
    def get_weight_options() -> dict:
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
    def plot_graph_edges(image: MatLike, nx_graph: nx.Graph, node_distribution_data: list = None, plot_nodes: bool = False, show_node_id: bool = False, add_width_thickness: bool = False, transparent: bool = False, edge_color: str= 'r', node_marker_size: float = 3) -> dict:
        """
        Plot graph edges on top of the image

        :param image: image to be superimposed with graph edges;
        :param nx_graph: a NetworkX graph;
        :param node_distribution_data: a list of node distribution data for a heatmap plot;
        :param plot_nodes: whether to plot graph nodes or not;
        :param show_node_id: if True, node IDs are displayed on the plot;
        :param add_width_thickness: whether to add width thickness to node distribution data;
        :param transparent: whether to draw the image with a transparent background;
        :param edge_color: each edge's line color;
        :param node_marker_size: the size (diameter) of the node marker
        :return:
        """

        def plot_graph_nodes(node_ax):
            """
            Plot graph nodes on top of the image.
            :param node_ax: Matplotlib axes
            """

            node_list = list(nx_graph.nodes())
            gn = np.array([nx_graph.nodes[i]['o'] for i in node_list])

            if show_node_id:
                i = 0
                for x, y in zip(gn[:, coord_1], gn[:, coord_2]):
                    node_ax.annotate(str(i), (x, y), fontsize=5)
                    i += 1

            if node_distribution_data is not None:
                c_set = node_ax.scatter(gn[:, coord_1], gn[:, coord_2], s=node_marker_size, c=node_distribution_data, cmap='plasma')
                return c_set
            else:
                # c_set = node_ax.scatter(gn[:, coord_1], gn[:, coord_2], s=marker_size)
                node_ax.plot(gn[:, coord_1], gn[:, coord_2], 'b.', markersize=node_marker_size)
                return None

        def create_plt_axes(pos) -> plt.Figure:
            """
            Create a matplotlib axes object.
            Args:
                pos: index position of image frame.

            Returns:

            """
            new_fig = plt.Figure()
            new_ax = new_fig.add_axes((0, 0, 1, 1))  # span the whole figure
            new_ax.set_axis_off()

            if image is None:
                return new_fig

            if transparent:
                new_ax.imshow(image[pos], cmap='gray', alpha=0)  # Alpha=0 makes image 100% transparent
            else:
                new_ax.imshow(image[pos], cmap='gray')
            return new_fig

        def normalize_width(w:float|numbers.Real|dict, new_min=0.5, new_max=5.0) -> float:
            if max_w == min_w:
                return (new_min + new_max) / 2  # avoid division by zero
            if not isinstance(w, numbers.Real):
                print(f"Invalid width type ({type(w)}); using default normalized width = 1.0")
                return 1.0
            norm_w = new_min + (w - min_w) * (new_max - new_min) / (max_w - min_w)
            return float(norm_w)

        def get_max_dims():
            xs, ys = [], []
            node_list = list(nx_graph.nodes())
            for i in node_list:
                pts = nx_graph.nodes[i].get("pts", None)
                if pts is not None:
                    pts = np.array(pts)
                    if pts.ndim == 1:  # single point [x, y]
                        xs.append(pts[0])
                        ys.append(pts[1])
                    elif pts.ndim == 2:  # multiple points [[x,y], [x,y], ...]
                        xs.extend(pts[:, 0])
                        ys.extend(pts[:, 1])

            if not xs or not ys:
                return None, None  # no "pts" found

            max_x = max(xs)
            max_y = max(ys)
            return max_x, max_y

        fig_group = {0: create_plt_axes(0)}
        if image is None:
            img_w, img_h = get_max_dims()
            if img_w is not None and img_h is not None:
                # GSD file has image dimensions but no image data
                image = np.ones((img_w, img_h), dtype=np.uint8) * 255
                image = [image]
            else:
                # Draw graph using NetworkX library
                ax = fig_group[0].get_axes()[0]
                if node_distribution_data is None:
                    # Planar: tries to avoid edge crossings, (working only for planar graphs?)
                    nx.draw(nx_graph, ax=ax, with_labels=show_node_id, node_size=node_marker_size, edge_color=edge_color)
                else:
                    # Normalize values for colormap
                    v_min, v_max = min(node_distribution_data), max(node_distribution_data)
                    nx.draw(nx_graph, ax=ax, with_labels=show_node_id,
                                   node_size=node_marker_size, node_color=node_distribution_data, cmap='plasma',
                                   vmin=v_min, vmax=v_max, edge_color=edge_color)
                    # Add colorbar for heatmap
                    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=v_min, vmax=v_max))
                    sm.set_array([])  # required for colorbar
                    cbar = fig_group[0].colorbar(sm, ax=ax, orientation='vertical', label='Value')
                    cbar.ax.set_position([0.82, 0.05, 0.05, 0.9])
                return fig_group

        # First, extract all widths to compute min and max
        all_widths = np.array([nx_graph[s][e]['width'] for s, e in nx_graph.edges()])
        if all_widths.size == 0:
            return fig_group
        min_w, max_w = min(all_widths), max(all_widths)

        # Create a color cycle for each graph component
        if edge_color == 'black':
            color_list = ['k', 'r', 'g', 'b', 'c', 'm', 'y']
        else:
            color_list = ['r', 'y', 'g', 'b', 'c', 'm', 'k']
        color_cycle = itertools.cycle(color_list)
        nx_components = list(nx.connected_components(nx_graph))
        for component in nx_components:
            color = next(color_cycle)
            sg = nx_graph.subgraph(component)

            for (s, e) in sg.edges():
                ge = sg[s][e]['pts']
                edge_w = 0.8
                if add_width_thickness:
                    wt = sg[s][e]['width']
                    edge_w = normalize_width(wt)  # The size of the plot line-width depends on width of edge
                coord_1, coord_2 = 1, 0  # coordinates: (y, x)
                coord_3 = 0
                if np.array(ge).shape[1] == 3:
                    # image and graph are 3D (not 2D)
                    # 3D Coordinates are (x, y, z) ... assume that y and z are the same for 2D graphs and x is depth.
                    coord_1, coord_2, coord_3 = 2, 1, 0  # coordinates: (z, y, x)

                if coord_3 in fig_group and fig_group[coord_3] is not None:
                    pass
                else:
                    fig_group[coord_3] = create_plt_axes(coord_3)
                ax = fig_group[coord_3].get_axes()[0]
                ax.plot(ge[:, coord_1], ge[:, coord_2], color, linewidth=edge_w)

        if plot_nodes:
            for idx, plt_fig in fig_group.items():
                ax = plt_fig.get_axes()[0]
                node_color_set = plot_graph_nodes(ax)
                if node_color_set is not None:
                    cbar = plt_fig.colorbar(node_color_set, ax=ax, orientation='vertical', label='Value')
                    # [left, bottom, width, height]
                    cbar.ax.set_position([0.82, 0.05, 0.05, 0.9])
        return fig_group
