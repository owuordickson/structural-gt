# SPDX-License-Identifier: GNU GPL v3

"""
Builds a graph network from nano-scale microscopy images.
"""

import csv
import cv2
import os
import math
import sknw
import itertools
import numpy as np
import scipy as sp
import networkx as nx
from ypstruct import struct
import matplotlib.pyplot as plt
from .progress_update import ProgressUpdate
from .graph_skeleton import GraphSkeleton
from .image_processor import ImageProcessor


class GraphConverter(ProgressUpdate):
    """
    A class for builds a graph network from microscopy images and stores is as a NetworkX object.

    :param img_obj: ImageProcessor object.
    :param options_gte: graph extraction parameters and options.
    """

    def __init__(self, img_obj: ImageProcessor, options_gte: struct = None):
        """
        A class for builds a graph network from microscopy images and stores is as a NetworkX object.

        :param img_obj: ImageProcessor object.
        :param options_gte: graph extraction parameters and options.

        >>> from ypstruct import struct
        >>> opt_img = struct()
        >>> opt_img.threshold_type = 1
        >>> opt_img.threshold_global = 127
        >>> opt_img.threshold_adaptive = 11
        >>> opt_img.gamma = float(1)
        >>> opt_img.gaussian_blurring_size = 3
        >>> opt_img.autolevel_blurring_size = 3
        >>> opt_img.lowpass_window_size = 10
        >>> opt_img.laplacian_kernel_size = 3
        >>> opt_img.sobel_kernel_size = 3
        >>> opt_img.apply_autolevel = 0
        >>> opt_img.apply_laplacian = 0
        >>> opt_img.apply_scharr = 0
        >>> opt_img.apply_sobel = 0
        >>> opt_img.apply_median = 0
        >>> opt_img.apply_gaussian = 0
        >>> opt_img.apply_lowpass = 0
        >>> opt_img.apply_dark_foreground = 0
        >>> opt_img.brightness_level = 0
        >>> opt_img.contrast_level = 0
        >>>
        >>> opt_gte = struct()
        >>> opt_gte.merge_nearby_nodes = 1
        >>> opt_gte.prune_dangling_edges = 1
        >>> opt_gte.remove_disconnected_segments = 1
        >>> opt_gte.remove_self_loops = 1
        >>> opt_gte.remove_object_size = 500
        >>> opt_gte.is_multigraph = 0
        >>> opt_gte.has_weights = 0
        >>> opt_gte.display_node_id = 0
        >>> opt_gte.export_edge_list = 0
        >>> opt_gte.export_as_gexf = 0
        >>> opt_gte.export_adj_mat = 0
        >>> opt_gte.save_images = 0
        >>>
        >>> i_path = "path/to/image"
        >>> o_dir = ""
        >>>
        >>> imp_obj = ImageProcessor(i_path, o_dir, options_img=opt_img)
        >>> graph_obj = GraphConverter(imp_obj, options_gte=opt_gte)
        >>> graph_obj.fit()
        """
        super(GraphConverter, self).__init__()
        self.terminal_app = True
        self.configs_graph = options_gte
        self.imp = img_obj
        self.graph_skeleton = None
        self.nx_graph, self.nx_info = None, []
        self.nx_components, self.nx_connected_graph, self.connect_ratio = [], None, 0

    def fit(self):
        """
        Execute functions that process image and builds a NetworkX graph from the image.

        :return:
        """
        self.update_status([10, "Processing image..."])
        self.imp.apply_filters()
        self.fit_graph()

    def fit_graph(self):
        """
        Execute a function that builds a NetworkX graph from the image.

        :return:
        """
        self.update_status([50, "Making graph skeleton..."])
        success = self.extract_graph()
        if not success:
            self.update_status([-1, "Problem encountered, provide GT parameters"])
        elif self.abort:
            self.update_status([-1, "Task aborted."])
        else:
            self.update_status([75, "Verifying graph network..."])
            if self.nx_graph.number_of_nodes() <= 0:
                self.update_status([-1, "Problem generating graph (change filter options)."])
            else:
                # self.save_adj_csv()
                # if self.configs_graph.has_weights == 1:
                #    self.nx_info, self.nx_components, self.connect_ratio = GraphComponents.compute_conductance(
                #        self.nx_graph, weighted=True)
                # else:
                self.nx_info, self.nx_components, self.connect_ratio = GraphComponents.compute_conductance(self.nx_graph)

                # draw graph network
                self.update_status([90, "Drawing graph network..."])
                graph_plt = self.draw_graph_network()
                self.imp.img_net = ImageProcessor.plot_to_img(graph_plt)

    def reset(self):
        """
        Erase the existing data stored in the object.
        :return:
        """
        self.imp.img_mod, self.imp.img_bin, self.imp.img_net = None, None, None
        self.nx_graph, self.nx_info = None, []

    def extract_graph(self):
        """
        Build a skeleton from image and use the skeleton to build a NetworkX graph.

        :return:
        """

        configs = self.configs_graph
        if configs is None:
            return False
        graph_skel = GraphSkeleton(self.imp.img_bin, configs)
        img_skel = graph_skel.skeleton
        self.graph_skeleton = graph_skel

        self.update_status([60, "Creating graph network..."])
        # skeleton analysis object with sknw
        if configs.is_multigraph:
            nx_graph = sknw.build_sknw(img_skel, multi=True)
            for (s, e) in nx_graph.edges():
                for k in range(int(len(nx_graph[s][e]))):
                    nx_graph[s][e][k]['length'] = nx_graph[s][e][k]['weight']
                    if nx_graph[s][e][k]['weight'] == 0:
                        nx_graph[s][e][k]['length'] = 2
            # since the skeleton is already built by skel_ID.py the weight that sknw finds will be the length
            # if we want the actual weights we get it from GetWeights.py, otherwise we drop them
            for (s, e) in nx_graph.edges():
                if configs.has_weights == 1:
                    for k in range(int(len(nx_graph[s][e]))):
                        ge = nx_graph[s][e][k]['pts']
                        pix_width, wt = graph_skel.assign_weights_by_width(ge)
                        nx_graph[s][e][k]['width'] = pix_width
                        nx_graph[s][e][k]['weight'] = wt
                else:
                    for k in range(int(len(nx_graph[s][e]))):
                        try:
                            del nx_graph[s][e][k]['weight']
                        except KeyError:
                            pass
        else:
            nx_graph = sknw.build_sknw(img_skel)
            for (s, e) in nx_graph.edges():
                # 'sknw' library stores length of edge and calls it weight, we reverse this
                # we create a new attribute 'length', later delete/modify 'weight'
                nx_graph[s][e]['length'] = nx_graph[s][e]['weight']
                #    if nx_graph[s][e]['weight'] == 0:  # TO BE DELETED later
                #        nx_graph[s][e]['length'] = 2

                if configs.has_weights == 1:
                    # We modify 'weight'
                    wt_type = configs.weight_type
                    px_size = self.imp.pixel_width
                    rho_val = self.imp.configs_img.resistivity
                    weight_options = GraphConverter.get_weight_options()

                    ge = nx_graph[s][e]['pts']
                    pix_width, pix_angle, wt = graph_skel.assign_weights(ge, wt_type, weight_options=weight_options,
                                                              pixel_dim=px_size, rho_dim=rho_val)
                    nx_graph[s][e]['width'] = pix_width
                    nx_graph[s][e]['angle'] = pix_angle
                    nx_graph[s][e]['weight'] = wt
                else:
                    ge = nx_graph[s][e]['pts']
                    pix_width, pix_angle, wt = graph_skel.assign_weights(ge, None)
                    nx_graph[s][e]['width'] = pix_width
                    nx_graph[s][e]['angle'] = pix_angle
                    nx_graph[s][e]['weight'] = wt
                    # delete 'weight'
                    del nx_graph[s][e]['weight']
                print(f"{nx_graph[s][e]}\n")
        self.nx_graph = nx_graph

        # Removing all instances of edges were the start and end are the same, or "self loops"
        if configs.remove_self_loops:
            if configs.is_multigraph:
                for (s, e) in list(self.nx_graph.edges()):
                    if s == e:
                        self.nx_graph.remove_edge(s, e)
            else:
                for (s, e) in self.nx_graph.edges():
                    if s == e:
                        self.nx_graph.remove_edge(s, e)
        return True

    def draw_graph_network(self, a4_size: bool = False, blank: bool = False):
        """
        Creates a plot figure of the graph network. It draws all the edges and nodes of the graph.

        :param a4_size: decision if to create an A4 size plot figure.
        :param blank: do not add image in the background, have a white background.

        :return:
        """

        opt_gte = self.configs_graph
        nx_graph = self.nx_graph
        nx_components = self.nx_components
        raw_img = self.imp.img

        if blank:
            w, h = raw_img.shape
            my_dpi = 96
            fig = plt.Figure(figsize=(h / my_dpi, w / my_dpi), dpi=my_dpi)
            ax = fig.add_axes((0, 0, 1, 1))  # span the whole figure
            ax.axis('off')
            ax.imshow(raw_img, cmap='gray', alpha=0)
            for (s, e) in nx_graph.edges():
                ge = nx_graph[s][e]['pts']
                ax.plot(ge[:, 1], ge[:, 0], 'black')
            nodes = nx_graph.nodes()
            gn = np.array([nodes[i]['o'] for i in nodes])
            ax.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)

            img = np.array(ImageProcessor.plot_to_img(fig))
            if len(img.shape) == 3:
                img = np.mean(img[:, :, :2], 2)  # Convert the image to grayscale (or 2D)
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
            ax.imshow(raw_img, cmap='gray')
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
            ax.imshow(raw_img, cmap='gray')
            if opt_gte.is_multigraph:
                for (s, e) in nx_graph.edges():
                    for k in range(int(len(nx_graph[s][e]))):
                        ge = nx_graph[s][e][k]['pts']
                        ax.plot(ge[:, 1], ge[:, 0], 'red')
            else:
                for (s, e) in nx_graph.edges():
                    ge = nx_graph[s][e]['pts']
                    ax.plot(ge[:, 1], ge[:, 0], 'red')

        return fig

    def display_skeletal_images(self):
        """
        Create plot figures of skeletal image and graph network image.

        :return:
        """

        opt_gte = self.configs_graph
        nx_graph = self.nx_graph
        g_skel = self.graph_skeleton
        img = self.imp.img

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
        ax_2.imshow(img, cmap='gray')
        if opt_gte.is_multigraph:
            for (s, e) in nx_graph.edges():
                for k in range(int(len(nx_graph[s][e]))):
                    ge = nx_graph[s][e][k]['pts']
                    ax_2.plot(ge[:, 1], ge[:, 0], 'red')
        else:
            for (s, e) in nx_graph.edges():
                ge = nx_graph[s][e]['pts']
                ax_2.plot(ge[:, 1], ge[:, 0], 'red')
        nodes = nx_graph.nodes()
        gn = np.array([nodes[i]['o'] for i in nodes])
        if opt_gte.display_node_id == 1:
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

        opt_gte = self.configs_graph

        run_info = "***Graph Extraction Configurations***\n"
        if opt_gte.has_weights:
            run_info += f"Weight Type: {GraphConverter.get_weight_options().get(opt_gte.weight_type)} || "
        if opt_gte.merge_nearby_nodes:
            run_info += "Merge Nodes || "
        if opt_gte.prune_dangling_edges:
            run_info += "Prune Dangling Edges || "
        run_info = run_info[:-3] + '' if run_info.endswith('|| ') else run_info
        run_info += "\n"
        if opt_gte.remove_disconnected_segments:
            run_info += f"Remove Objects of Size = {opt_gte.remove_object_size} || "
        if opt_gte.remove_self_loops:
            run_info += "Remove Self Loops || "
        if opt_gte.is_multigraph:
            run_info += "Multi-graph allowed "
        run_info = run_info[:-3] + '' if run_info.endswith('|| ') else run_info

        return run_info

    def save_files(self, opt_gte: struct = None):
        """
        Save graph data into files.

        :param opt_gte:
        :return:
        """

        nx_graph = self.nx_graph.copy()
        if opt_gte is None:
            opt_gte = self.configs_graph
        filename, output_location = self.imp.create_filenames()
        g_filename = filename + "_graph.gexf"
        el_filename = filename + "_EL.csv"
        adj_filename = filename + "_adj.csv"
        pr_filename = filename + "_processed.jpg"
        bin_filename = filename + "_binary.jpg"
        net_filename = filename + "_final.jpg"
        gexf_file = os.path.join(output_location, g_filename)
        csv_file = os.path.join(output_location, el_filename)
        adj_file = os.path.join(output_location, adj_filename)
        img_file = os.path.join(output_location, pr_filename)
        bin_file = os.path.join(output_location, bin_filename)
        net_file = os.path.join(output_location, net_filename)

        if opt_gte.save_images == 1:
            graph_img = self.imp.img_net
            cv2.imwrite(str(img_file), self.imp.img_mod)
            cv2.imwrite(str(bin_file), self.imp.img_bin)
            if graph_img.mode == "JPEG":
                graph_img.save(net_file, format='JPEG', quality=95)
            elif graph_img.mode in ["RGBA", "P"]:
                img_net = graph_img.convert("RGB")
                img_net.save(net_file, format='JPEG', quality=95)

        if opt_gte.export_adj_mat == 1:
            adj_mat = nx.adjacency_matrix(self.nx_graph).todense()
            np.savetxt(str(adj_file), adj_mat, delimiter=",")

        if opt_gte.export_edge_list == 1:
            if opt_gte.has_weights == 1:
                fields = ['Source', 'Target', 'Weight', 'Length']
                el = nx.generate_edgelist(nx_graph, delimiter=',', data=["weight", "length"])
                with open(csv_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(fields)
                    for line in el:
                        line = str(line)
                        row = line.split(',')
                        try:
                            writer.writerow(row)
                        except csv.Error:
                            pass
                csvfile.close()
            else:
                fields = ['Source', 'Target']
                el = nx.generate_edgelist(nx_graph, delimiter=',', data=False)
                with open(csv_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(fields)
                    for line in el:
                        line = str(line)
                        row = line.split(',')
                        try:
                            writer.writerow(row)
                        except csv.Error:
                            pass
                csvfile.close()

        if opt_gte.export_as_gexf == 1:
            if opt_gte.is_multigraph:
                # deleting extraneous info and then exporting the final skeleton
                for (x) in nx_graph.nodes():
                    del nx_graph.nodes[x]['pts']
                    del nx_graph.nodes[x]['o']
                for (s, e) in nx_graph.edges():
                    for k in range(int(len(nx_graph[s][e]))):
                        try:
                            del nx_graph[s][e][k]['pts']
                        except KeyError:
                            pass
                nx.write_gexf(nx_graph, gexf_file)
            else:
                # deleting extraneous info and then exporting the final skeleton
                for (x) in nx_graph.nodes():
                    del nx_graph.nodes[x]['pts']
                    del nx_graph.nodes[x]['o']
                for (s, e) in nx_graph.edges():
                    del nx_graph[s][e]['pts']
                nx.write_gexf(nx_graph, gexf_file)

    def compute_edge_length(self):
        pass

    def compute_edge_width(self):
        pass

    def compute_edge_angle(self):
        pass

    @staticmethod
    def get_weight_options():
        """
        Returns the weight options for building the graph edges.

        :return:
        """
        weight_options = {
            'DIA': 'Width',
            'AREA': 'Area',  # surface area of edge
            'LEN': 'Length',
            'INV_LEN': 'InverseLength',
            'VAR_CON': 'Conductance',  # with variable width
            'FIX_CON': 'FixedWidthConductance',
            'RES': 'Resistance',
            # '': ''
        }
        return weight_options


class GraphComponents:

    @staticmethod
    def compute_conductance(graph: nx.Graph, weighted: bool = False):
        """
        Computes graph conductance through an approach based on eigenvectors or spectral frequency.\
        Implements ideas proposed in:    https://doi.org/10.1016/j.procs.2013.09.311.

        Conductance can closely be approximated via eigenvalue computation,\
        a fact which has been well-known and well-used in the graph theory community.

        The Laplacian matrix of a directed graph is by definition generally non-symmetric,\
        while, e.g., traditional spectral clustering is primarily developed for undirected\
        graphs with symmetric adjacency and Laplacian matrices. A trivial approach to apply\
        techniques requiring the symmetry is to turn the original directed graph into an\
        undirected graph and build the Laplacian matrix for the latter.

        We need to remove isolated nodes (in order to avoid singular adjacency matrix).\
        The degree of a node is the number of edges incident to that node.\
        When a node has a degree of zero, it means that there are no edges\
        connected to that node. In other words, the node is isolated from\
        the rest of the graph.

        :param weighted: is graph a weighted graph?
        :param graph: NetworkX graph

        """

        # It is important to notice our graph is (mostly) a directed graph,
        # meaning that it is: (asymmetric) with self-looping nodes

        data = []
        # sub_components = []

        # 1. Make a copy of the graph
        # eig_graph = graph.copy()

        # 2a. Remove self-looping edges
        eig_graph = GraphComponents.remove_self_loops(graph)

        # 2b. Identify isolated nodes
        isolated_nodes = list(nx.isolates(eig_graph))

        # 2c. Remove isolated nodes
        eig_graph.remove_nodes_from(isolated_nodes)

        # 3a. Check connectivity of graph
        # It has less than two nodes or is not connected.
        sub_graph_largest, sub_graph_smallest, size, sub_components = GraphComponents.get_graph_components(
                eig_graph)
        eig_graph = sub_graph_largest
        if size > 1:
            data.append({"name": "Subgraph Count", "value": size})
            data.append({"name": "Large Subgraph Node Count", "value": sub_graph_largest.number_of_nodes()})
            data.append({"name": "Large Subgraph Edge Count", "value": sub_graph_largest.number_of_edges()})
            data.append({"name": "Small Subgraph Node Count", "value": sub_graph_smallest.number_of_nodes()})
            data.append({"name": "Small Subgraph Edge Count", "value": sub_graph_smallest.number_of_edges()})

        # 4. Compute normalized-laplacian matrix
        if weighted:
            norm_laplacian_matrix = nx.normalized_laplacian_matrix(eig_graph, weight='weight').toarray()
        else:
            # norm_laplacian_matrix = compute_norm_laplacian_matrix(eig_graph)
            norm_laplacian_matrix = nx.normalized_laplacian_matrix(eig_graph).toarray()

        # 5. Compute eigenvalues
        # e_vals, _ = np.linalg.eig(norm_laplacian_matrix)
        e_vals = sp.linalg.eigvals(norm_laplacian_matrix)

        # 6. Approximate conductance using the 2nd smallest eigenvalue
        eigenvalues = e_vals.real
        val_max, val_min = GraphComponents.compute_conductance_range(eigenvalues)
        data.append({"name": "Graph Conductance (max)", "value": val_max})
        data.append({"name": "Graph Conductance (min)", "value": val_min})
        ratio = eig_graph.number_of_nodes() / graph.number_of_nodes()

        return data, sub_components, ratio

    @staticmethod
    def remove_self_loops(graph: nx.Graph):
        """
        Remove self-loops from graph, they cause zero values in Degree matrix.

        :param graph:
        :return:
        """

        # 1. Get Adjacency matrix
        adj_mat = nx.adjacency_matrix(graph).todense()

        # 2. Symmetric-ize the Adjacency matrix
        # adj_mat = np.maximum(adj_mat, adj_mat.transpose())

        # 3. Remove (self-loops) non-zero diagonal values in Adjacency matrix
        np.fill_diagonal(adj_mat, 0)

        # 4. Create new graph
        new_graph = nx.from_numpy_array(adj_mat)

        return new_graph

    @staticmethod
    def make_graph_symmetrical(graph: nx.Graph):
        """
        Deletes diagonal items to make the adjacency matrix of a graph symmetrical. It removes self-loops.

        :param graph:
        :return:
        """

        # 1. Get Adjacency matrix
        adj_mat = nx.adjacency_matrix(graph).todense()

        # 2. Symmetric-ize the Adjacency matrix
        adj_mat = np.maximum(adj_mat, adj_mat.transpose())

        # 3. Remove (self-loops) non-zero diagonal values in Adjacency matrix
        np.fill_diagonal(adj_mat, 0)

        # 4. Create new graph
        new_graph = nx.from_numpy_array(adj_mat)

        return new_graph

    @staticmethod
    def compute_conductance_range(eig_vals: np.ndarray):
        """
        Computes the minimum and maximum values of graph conductance.

        :param eig_vals:
        :return:
        """

        # Sort the eigenvalues in ascending order
        sorted_vals = np.array(eig_vals)
        sorted_vals.sort()

        # Sort the eigenvalues in descending order
        # eigenvalues[::-1].sort()

        # approximate conductance using the 2nd smallest eigenvalue
        try:
            conductance_max = math.sqrt((2 * sorted_vals[1]))
        except ValueError:
            conductance_max = None
        conductance_min = sorted_vals[1] / 2

        return conductance_max, conductance_min

    @staticmethod
    def get_graph_components(graph: nx.Graph):
        """
        Retrieves the subcomponents that make up the entire NetworkX graph.

        :param graph:
        :return:
        """

        # 1. Identify connected components
        connected_components = list(nx.connected_components(graph))

        # 2. Find the largest/smallest connected component
        largest_component = max(connected_components, key=len)
        smallest_component = min(connected_components, key=len)

        # 3. Create a new graph containing only the largest/smallest connected component
        sub_graph_largest = graph.subgraph(largest_component)
        sub_graph_smallest = graph.subgraph(smallest_component)

        component_count = len(connected_components)
        # large_subgraph_node_count = sub_graph_largest.number_of_nodes()
        # small_subgraph_node_count = sub_graph_smallest.number_of_nodes()
        # large_subgraph_edge_count = sub_graph_largest.number_of_edges()
        # small_subgraph_edge_count = sub_graph_smallest.number_of_edges()

        return sub_graph_largest, sub_graph_smallest, component_count, connected_components
