# SPDX-License-Identifier: GNU GPL v3

"""
Builds a graph network from nanoscale microscopy images.
"""

import os
import igraph
import sknw
import logging
import numpy as np
import networkx as nx
from PIL import Image, ImageQt
from cv2.typing import MatLike
from ovito.data import DataCollection, Particles
from ovito.pipeline import StaticSource, Pipeline
from ovito.vis import Viewport
import matplotlib.pyplot as plt

from .progress_update import ProgressUpdate
from .graph_skeleton import GraphSkeleton
from .sgt_utils import write_csv_file, write_gsd_file, plot_to_opencv
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


class FiberNetworkBuilder(ProgressUpdate):
    """
    A class for builds a graph network from microscopy images and stores is as a NetworkX object.

    """

    def __init__(self):
        """
        A class for builds a graph network from microscopy images and stores is as a NetworkX object.

        """
        super(FiberNetworkBuilder, self).__init__()
        self.configs: dict = load_gte_configs()  # graph extraction parameters and options.
        self.props: list = []
        self.img_ntwk: MatLike | None = None
        self.nx_3d_graph: nx.Graph | None = None
        self.ig_graph: igraph.Graph | None = None
        self.gsd_file: str | None = None
        self.skel_obj: GraphSkeleton | None = None

    def fit_graph(self, save_dir: str, img_bin: MatLike = None, img_2d: MatLike = None, is_img_2d: bool = True, px_width_sz: float = 1.0, rho_val: float = 1.0, image_file: str = "img"):
        """
        Execute a function that builds a NetworkX graph from the binary image.

        :param save_dir: Directory to save the graph to.
        :param img_bin: A binary image for building Graph Skeleton for the NetworkX graph.
        :param img_2d: The actual 2D image(s) used for plotting the graph network.
        :param is_img_2d: Whether the image is 2D or 3D otherwise.
        :param px_width_sz: Width of a pixel in nanometers.
        :param rho_val: Resistivity coefficient/value of the material.
        :param image_file: Filename of the binary image.
        :return:
        """

        if self.abort:
            self.update_status([-1, "Task aborted by due to an error. If problem with graph: change/apply different "
                                    "image/binary filters and graph options. OR change brightness/contrast"])
            return

        self.update_status([50, "Extracting the graph network..."])
        success = self.extract_graph(image_bin=img_bin, is_img_2d=is_img_2d, px_size=px_width_sz, rho_val=rho_val)
        if not success:
            self.update_status([-1, "Problem encountered, provide GT parameters"])
            self.abort = True
            return

        self.update_status([75, "Verifying graph network..."])
        if self.nx_3d_graph.number_of_nodes() <= 0:
            self.update_status([-1, "Problem generating graph (change image/binary filters)"])
            self.abort = True
            return

        self.update_status([77, "Retrieving graph properties..."])
        self.props = self.get_graph_props()

        self.update_status([90, "Saving graph network..."])
        # Save graph to GSD/HOOMD - For OVITO rendering
        self.configs["export_as_gsd"]["value"] = 1
        self.save_graph_to_file(image_file, save_dir)

        self.update_status([95, "Plotting graph network..."])
        plt_fig = self.plot_2d_graph_network(img_2d, is_img_2d)
        self.img_ntwk = plot_to_opencv(plt_fig)

    def reset_graph(self):
        """
        Erase the existing data stored in the object.
        :return:
        """
        self.nx_3d_graph, self.ig_graph, self.img_ntwk = None, None, None

    def extract_graph(self, image_bin: MatLike = None, is_img_2d: bool = True, px_size: float = 1.0, rho_val: float = 1.0):
        """
        Build a skeleton from the image and use the skeleton to build a NetworkX graph.

        :param image_bin: Binary image from which the skeleton will be built and graph drawn.
        :param is_img_2d: Whether the image is 2D or 3D otherwise.
        :param px_size: Width of a pixel in nanometers.
        :param rho_val: Resistivity coefficient/value of the material.
        :return:
        """

        if image_bin is None:
            return False

        opt_gte = self.configs
        if opt_gte is None:
            return False

        self.update_status([58, "Build graph skeleton from binary image..."])
        graph_skel = GraphSkeleton(image_bin, opt_gte, is_2d=is_img_2d, progress_func=self.update_status)
        self.skel_obj = graph_skel
        img_skel = graph_skel.skeleton  # .astype(int)

        self.update_status([60, "Creating graph network..."])
        nx_graph = sknw.build_sknw(img_skel)

        if opt_gte["remove_self_loops"]["value"]:
            self.update_status([64, "Removing self loops from graph network..."])

        self.update_status([66, "Assigning weights to graph network..."])
        for (s, e) in nx_graph.edges():

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
            else:
                pix_width, pix_angle, wt = graph_skel.assign_weights(ge, None)
                del nx_graph[s][e]['weight']            # delete 'weight'
            nx_graph[s][e]['width'] = pix_width
            nx_graph[s][e]['angle'] = pix_angle
            nx_graph[s][e]['weight'] = wt
            # print(f"{nx_graph[s][e]}\n")
        self.nx_3d_graph = nx_graph
        self.ig_graph = igraph.Graph.from_networkx(nx_graph)
        return True

    def plot_2d_graph_network(self, image_2d_arr: MatLike, plot_nodes: bool = False, a4_size: bool = False):
        """
        Creates a plot figure of the graph network. It draws all the edges and nodes of the graph.

        :param image_2d_arr: Slides of 2D images to be used to draw the network.
        :param plot_nodes: Make the graph's node plot figure.
        :param a4_size: Decision if to create an A4 size plot figure.

        :return:
        """

        # Fetch the graph and config options
        nx_graph = self.nx_3d_graph
        is_img_2d = self.skel_obj.is_2d
        show_node_id = (self.configs["display_node_id"]["value"] == 1)

        # Fetch a single 2D image
        if image_2d_arr is None:
            return None
        else:
            image_2d = image_2d_arr[0]

        # Create the plot figure
        if a4_size:
            fig = plt.Figure(figsize=(8.5, 11), dpi=400)
            ax = fig.add_subplot(1, 1, 1)
            plt_title = "Graph Node Plot" if plot_nodes else "Graph Edge Plot"
            ax.set_title(plt_title)
        else:
            fig = plt.Figure()
            ax = fig.add_axes((0, 0, 1, 1))  # span the whole figure
        FiberNetworkBuilder.plot_graph_edges(ax, image_2d, nx_graph, is_graph_2d=is_img_2d, color='yellow')
        if plot_nodes:
            FiberNetworkBuilder.plot_graph_nodes(ax, nx_graph, is_graph_2d=is_img_2d, display_node_id=show_node_id)
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

    def get_graph_props(self):
        """
        A method that retrieves graph properties and stores them in a list-array.

        Returns: list of graph properties
        """

        # 1. Identify the subcomponents (graph segments) that make up the entire NetworkX graph.
        self.update_status([78, "Identifying graph subcomponents..."])
        graph = self.nx_3d_graph.copy()
        connected_components = list(nx.connected_components(graph))
        if not connected_components:  # In case the graph is empty
            connected_components = []
        sub_graphs = [graph.subgraph(c).copy() for c in connected_components]
        giant_graph = max(sub_graphs, key=lambda g: g.number_of_nodes())
        num_graphs = len(sub_graphs)
        connect_ratio = giant_graph.number_of_nodes() / graph.number_of_nodes()

        # 2. Update with the giant graph
        self.nx_3d_graph = giant_graph

        # 3. Populate graph properties
        self.update_status([80, "Storing graph properties..."])
        props = [
            ["Weight Type", str(FiberNetworkBuilder.get_weight_options().get(self.get_weight_type()))],
            ["Edge Count", str(graph.number_of_edges())],
            ["Node Count", str(graph.number_of_nodes())],
            ["Graph Count", str(len(connected_components))],
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

        :param filename: The filename to save the data to.
        :param out_dir: The directory to save the data to.
        :return:
        """

        nx_graph = self.nx_3d_graph.copy()
        opt_gte = self.configs

        g_filename = filename + "_graph.gexf"
        el_filename = filename + "_EL.csv"
        adj_filename = filename + "_adj.csv"
        gsd_filename = filename + "_skel.gsd"
        gexf_file = os.path.join(out_dir, g_filename)
        csv_file = os.path.join(out_dir, el_filename)
        adj_file = os.path.join(out_dir, adj_filename)

        if opt_gte["export_adj_mat"]["value"] == 1:
            adj_mat = nx.adjacency_matrix(self.nx_3d_graph).todense()
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
            self.gsd_file = os.path.join(out_dir, gsd_filename)
            if self.skel_obj.skeleton_3d is not None:
                write_gsd_file(self.gsd_file, self.skel_obj.skeleton_3d)

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
    def plot_graph_edges(axis, image: MatLike, nx_graph: nx.Graph, is_graph_2d: bool, transparent: bool = False, line_width: float=1.5, color: str = 'black'):
        """
        Plot graph edges on top of the image.

        :param axis: Matplotlib axis
        :param image: image to be superimposed with graph edges
        :param nx_graph: a NetworkX graph
        :param is_graph_2d: whether the generated graph is 2D or 3D
        :param transparent: whether to draw the image with a transparent background
        :param line_width: each edge's line width
        :param color: each edge's color
        :return:
        """

        axis.set_axis_off()
        axis.imshow(image, cmap='gray')

        if is_graph_2d:
            coord_1, coord_2 = 1, 0         # coordinates: (y, x)
        else:
            coord_1, coord_2 = 2, 1         # coordinates: (z, y)

        if transparent:
            axis.imshow(image, cmap='gray', alpha=0)  # Alpha=0 makes image 100% transparent

        # DOES NOT PLOT 3D graphs - node_plot (BUG IN CODE), so we pick the first 2D graph in the stack
        for (s, e) in nx_graph.edges():
            # 3D Coordinates are (x, y, z) ... assume that y and z are the same for 2D graphs and x is depth.
            ge = nx_graph[s][e]['pts']
            axis.plot(ge[:, coord_1], ge[:, coord_2], color, linewidth=line_width)
        return axis

    @staticmethod
    def plot_graph_nodes(axis: plt.axis, nx_graph: nx.Graph, is_graph_2d: bool, marker_size: float = 3, distribution_data: list = None, display_node_id: bool = False):
        """
        Plot graph nodes on top of the image.
        :param axis: Matplotlib axis
        :param nx_graph: a NetworkX graph
        :param is_graph_2d: whether the generated graph is 2D or 3D
        :param marker_size: the size of each node
        :param distribution_data: the heatmap distribution data
        :param display_node_id: indicate the node id on the plot
        :return:

        """

        node_list = list(nx_graph.nodes())
        gn = xp.array([nx_graph.nodes[i]['o'] for i in node_list])
        # 3D Coordinates are (x, y, z) ... assume that y and z are the same for 2D graphs and x is depth.

        if is_graph_2d:
            coord_1, coord_2 = 1, 0         # coordinates: (y, x)
        else:
            coord_1, coord_2 = 2, 1         # coordinates: (z, y)

        if display_node_id:
            i = 0
            for x, y in zip(gn[:, coord_1], gn[:, coord_2]):
                axis.annotate(str(i), (x, y), fontsize=5)
                i += 1

        if distribution_data is not None:
            c_set = axis.scatter(gn[:, coord_1], gn[:, coord_2], s=marker_size, c=distribution_data, cmap='plasma')
            return c_set
        else:
            # c_set = axis.scatter(gn[:, coord_1], gn[:, coord_2], s=marker_size)
            axis.plot(gn[:, coord_1], gn[:, coord_2], 'b.', markersize=marker_size)
            return axis

    # TO DELETE IT LATER
    def data_gen_function(self):
        """Populates OVITO's data with particles."""
        data = DataCollection()
        positions = np.asarray(np.where(np.asarray(self.skel_obj.skeleton_3d) != 0)).T
        particles = Particles()
        particles.create_property("Position", data=positions)
        data.objects.append(particles)
        return data

    # ONLY RUNS ON MAIN THREAD - TO BE DELETED
    def render_graph_to_image(self, bg_image=None, is_img_2d: bool = True):
        """
        Renders the graph network into an image; it can optionally superimpose the graph on the image.

        :param bg_image: Optional background image.
        :param is_img_2d: Whether the image is 2D or 3D otherwise.
        """
        if self.gsd_file is None:
            return None

        if bg_image is not None:
            # OVITO doesn’t directly support 3D numpy volumes as backgrounds
            bg_image = bg_image.squeeze()
            if not is_img_2d:
                # (visualize only one slice) Extract a middle slice from 3D grayscale volume
                mid_slice = bg_image[bg_image.shape[0] // 2]  # shape: (H, W)
            else:
                mid_slice = bg_image

            # Convert to RGB for PIL
            # bg_rgb = cv2.cvtColor(mid_slice, cv2.COLOR_GRAY2RGB)
            bg_pil = Image.fromarray(mid_slice).convert("RGB")

            # Set OVITO render size
            size = (bg_pil.width, bg_pil.height)
        else:
            size = (800, 600)
            bg_pil = None

        # Load OVITO pipeline and scene
        # pipeline = import_file(self.gsd_file)
        data = self.data_gen_function()
        # print(type(data))
        pipeline = Pipeline(source = StaticSource(data = data))
        pipeline.add_to_scene()

        vp = Viewport(type=Viewport.Type.Front, camera_dir=(2, 1, -1))
        vp.zoom_all(size)

        # Render to QImage without the alpha channel
        q_img = vp.render_image(size=size, alpha=True, background=(1, 1, 1))

        # Convert QImage to PIL Image
        pil_img = ImageQt.fromqimage(q_img).convert("RGB")

        if bg_pil is not None:
            # Overlay using simple blending (optional: adjust transparency)
            final_img = Image.blend(bg_pil, pil_img, alpha=0.5)
        else:
            final_img = pil_img
        return final_img
