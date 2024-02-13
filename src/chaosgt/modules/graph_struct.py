# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Processing of images and chaos engineering
"""

import cv2
import re
import os
import io
import sknw
import multiprocessing
import numpy as np
import networkx as nx
# import porespy as ps
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import disk
from skimage.filters.rank import autolevel, median
from .graph_skeleton import GraphSkeleton


class GraphStruct:

    def __init__(self, img_path, out_path, options_img, options_gte):
        self.__listeners = []
        self.terminal_app = True
        self.configs_img = options_img
        self.configs_graph = options_gte
        self.output_path = out_path
        self.img_raw = GraphStruct.load_img_from_file(img_path)
        self.img = self.resize_img(512)
        self.img_path = img_path
        self.img_bin = None
        self.img_net = None
        self.img_plot = None
        self.img_filtered = None
        self.graph_skeleton = None
        self.nx_graph = None
        self.nx_connected_graph = None
        self.otsu_val = None
        self.connectedness_ratio = 0

    def add_listener(self, func):
        """
        Add functions from the list of listeners.
        :param func:
        :return:
        """
        if func in self.__listeners:
            return
        self.__listeners.append(func)

    def remove_listener(self, func):
        """
        Remove functions from the list of listeners.
        :param func:
        :return:
        """
        if func not in self.__listeners:
            return
        self.__listeners.remove(func)

    # Trigger events.
    def update_status(self, args=None):
        # Run all the functions that are saved.
        if args is None:
            args = []
        for func in self.__listeners:
            func(*args)

    def fit(self):
        self.update_status([10, "Processing image..."])
        self.img_filtered = self.process_img()
        self.img_bin, self.otsu_val = self.binarize_img(self.img_filtered.copy())
        self.update_status([50, "Extracting graph..."])
        self.extract_graph()
        self.update_status([75, "Verifying graph network..."])
        self.nx_connected_graph, self.connectedness_ratio = self.find_largest_subgraph()
        if self.nx_graph.number_of_nodes() <= 0:
            self.update_status([-1, "Problem generating graph (change filter options)."])
        else:
            # draw graph network
            self.update_status([90, "Drawing graph network..."])
            self.img_plot, self.img_net = self.draw_graph_network(self.configs_graph)

    def resize_img(self, size):
        w, h = self.img_raw.shape
        if h > w:
            scale_factor = size / h
        else:
            scale_factor = size / w
        std_width = int(scale_factor * w)
        std_height = int(scale_factor * h)
        std_size = (std_height, std_width)
        std_img = cv2.resize(self.img_raw, std_size)
        return std_img

    def process_img(self):
        """

        :return:
        """

        options = self.configs_img
        filtered_img = self.img.copy()

        brightness_val = ((options.brightness_level / 100) * 510) - 255
        contrast_val = ((options.contrast_level / 100) * 254) - 127
        filtered_img = GraphStruct.control_brightness(filtered_img, brightness_val, contrast_val)

        if options.gamma != 1.00:
            inv_gamma = 1.00 / options.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype('uint8')
            filtered_img = cv2.LUT(filtered_img, table)

        # applies a low-pass filter
        if options.apply_lowpass == 1:
            w, h = filtered_img.shape
            ham1x = np.hamming(w)[:, None]  # 1D hamming
            ham1y = np.hamming(h)[:, None]  # 1D hamming
            ham2d = np.sqrt(np.dot(ham1x, ham1y.T)) ** options.lowpass_window_size  # expand to 2D hamming
            f = cv2.dft(filtered_img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shifted = np.fft.fftshift(f)
            f_complex = f_shifted[:, :, 0] * 1j + f_shifted[:, :, 1]
            f_filtered = ham2d * f_complex
            f_filtered_shifted = np.fft.fftshift(f_filtered)
            inv_img = np.fft.ifft2(f_filtered_shifted)  # inverse F.T.
            filtered_img = np.abs(inv_img)
            filtered_img -= filtered_img.min()
            filtered_img = filtered_img * 255 / filtered_img.max()
            filtered_img = filtered_img.astype(np.uint8)

        # applying median filter
        if options.apply_median == 1:
            # making a 5x5 array of all 1's for median filter
            med_disk = disk(5)
            filtered_img = median(filtered_img, med_disk)

        # applying gaussian blur
        if options.apply_gaussian == 1:
            b_size = options.gaussian_blurring_size
            filtered_img = cv2.GaussianBlur(filtered_img, (b_size, b_size), 0)

        # applying auto-level filter
        if options.apply_autolevel == 1:
            # making a disk for the auto-level filter
            auto_lvl_disk = disk(options.autolevel_blurring_size)
            filtered_img = autolevel(filtered_img, footprint=auto_lvl_disk)

        # applying a scharr filter, and then taking that image and weighting it 25% with the original
        # this should bring out the edges without separating each "edge" into two separate parallel ones
        if options.apply_scharr == 1:
            d_depth = cv2.CV_16S
            grad_x = cv2.Scharr(filtered_img, d_depth, 1, 0)
            grad_y = cv2.Scharr(filtered_img, d_depth, 0, 1)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            dst = cv2.convertScaleAbs(dst)
            filtered_img = cv2.addWeighted(filtered_img, 0.75, dst, 0.25, 0)
            filtered_img = cv2.convertScaleAbs(filtered_img)

        # applying sobel filter
        if options.apply_sobel == 1:
            scale = 1
            delta = 0
            d_depth = cv2.CV_16S
            grad_x = cv2.Sobel(filtered_img, d_depth, 1, 0, ksize=options.sobel_kernel_size, scale=scale,
                               delta=delta, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(filtered_img, d_depth, 0, 1, ksize=options.sobel_kernel_size, scale=scale,
                               delta=delta, borderType=cv2.BORDER_DEFAULT)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            dst = cv2.convertScaleAbs(dst)
            filtered_img = cv2.addWeighted(filtered_img, 0.75, dst, 0.25, 0)
            filtered_img = cv2.convertScaleAbs(filtered_img)

        # applying laplacian filter
        if options.apply_laplacian == 1:
            d_depth = cv2.CV_16S
            dst = cv2.Laplacian(filtered_img, d_depth, ksize=options.laplacian_kernel_size)
            # dst = cv2.Canny(img_filtered, 100, 200); # canny edge detection test
            dst = cv2.convertScaleAbs(dst)
            filtered_img = cv2.addWeighted(filtered_img, 0.75, dst, 0.25, 0)
            filtered_img = cv2.convertScaleAbs(filtered_img)

        return filtered_img

    def binarize_img(self, image):
        """

        :return:
        """
        # image = self.img_filtered.copy()
        img_bin = None
        options = self.configs_img
        # only needed for OTSU threshold
        otsu_res = 0

        if image is None:
            return None

        # applying universal threshold, checking if it should be inverted (dark foreground)
        if options.threshold_type == 0:
            if options.apply_dark_foreground == 1:
                img_bin = cv2.threshold(image, options.threshold_global, 255, cv2.THRESH_BINARY_INV)[1]
            else:
                img_bin = cv2.threshold(image, options.threshold_global, 255, cv2.THRESH_BINARY)[1]

        # adaptive threshold generation
        elif options.threshold_type == 1:
            if options.apply_dark_foreground == 1:
                img_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, options.threshold_adaptive, 2)
            else:
                img_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, options.threshold_adaptive, 2)

        # OTSU threshold generation
        elif options.threshold_type == 2:
            if options.apply_dark_foreground == 1:
                temp = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                img_bin = temp[1]
                otsu_res = temp[0]
            else:
                temp = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img_bin = temp[1]
                otsu_res = temp[0]

        return img_bin, otsu_res

    def extract_graph(self):
        """

        :return:
        """

        configs = self.configs_graph
        graph_skel = GraphSkeleton(self.img_bin, configs)
        img_skel = graph_skel.skeleton
        self.graph_skeleton = graph_skel

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
                if configs.weighted_by_diameter == 1:
                    for k in range(int(len(nx_graph[s][e]))):
                        ge = nx_graph[s][e][k]['pts']
                        pix_width, wt = graph_skel.assign_weights_by_width(ge)
                        nx_graph[s][e][k]['pixel width'] = pix_width
                        nx_graph[s][e][k]['weight'] = wt
                else:
                    for k in range(int(len(nx_graph[s][e]))):
                        try:
                            del nx_graph[s][e][k]['weight']
                        except KeyError:
                            pass
            self.nx_graph = nx_graph
        else:
            nx_graph = sknw.build_sknw(img_skel)

            # the actual length of the edges we want is stored as weight, so the two are set equal
            # if the weight is 0 the edge length is set to 2
            # for (s, e) in nx_graph.edges():
            #    nx_graph[s][e]['length'] = nx_graph[s][e]['weight']
            #    if nx_graph[s][e]['weight'] == 0:
            #        nx_graph[s][e]['length'] = 2
            with multiprocessing.Pool() as pool:
                items_1 = [(nx_graph, s, e) for (s, e) in nx_graph.edges()]
                for graph in pool.starmap(GraphStruct._task_init_weight, items_1):
                    nx_graph = graph

            # since the skeleton is already built by skel_ID.py the weight that sknw finds will be the length
            # if we want the actual weights we get it from GetWeights.py, otherwise we drop them
            # for (s, e) in nx_graph.edges():
            #    if configs.weighted_by_diameter == 1:
            #        ge = nx_graph[s][e]['pts']
            #        pix_width, wt = graph_skel.assign_weights_by_width(ge)
            #        nx_graph[s][e]['pixel width'] = pix_width
            #        nx_graph[s][e]['weight'] = wt
            #    else:
            #        del nx_graph[s][e]['weight']
            with multiprocessing.Pool() as pool:
                items_2 = [(configs, graph_skel, nx_graph, s, e) for (s, e) in nx_graph.edges()]
                for graph in pool.starmap(GraphStruct._task_assign_weight, items_2):
                    nx_graph = graph

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

    def find_largest_subgraph(self):
        """

        :return:
        """

        # 1. Identify connected components
        largest, smallest, count, connected_components = GraphStruct.graph_components(self.nx_graph)

        if len(connected_components) <= 0:
            return self.nx_graph, 1
        else:
            # 2. Find the largest/smallest connected component
            # Create a new graph containing only the largest connected component
            # Compute proportion
            ratio = largest.number_of_nodes() / self.nx_graph.number_of_nodes()

            return largest, round(ratio, 4)

    def compute_fractal_dimension(self):
        self.update_status([-1, "Computing fractal dimension..."])
        # sierpinski_im = ps.generators.sierpinski_foam(4, 5)
        fd_metrics = ps.metrics.boxcount(self.img)
        print(fd_metrics.slope)
        x = np.log(np.array(fd_metrics.size))
        y = np.log(np.array(fd_metrics.count))
        fractal_dimension = np.polyfit(x, y, 1)[0]  # fractal_dimension = lim r -> 0 log(Nr)/log(1/r)
        print(fractal_dimension)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_xlabel('box size')
        ax1.set_ylabel('box count')
        ax2.set_xlabel('box size')
        ax2.set_ylabel('slope')
        ax2.set_xscale('log')
        ax1.plot(fd_metrics.size, fd_metrics.count, '-o')
        ax2.plot(fd_metrics.size, fd_metrics.slope, '-o')
        plt.show()

    def draw_graph_network(self, opt_gte):
        """

        :param opt_gte:
        :return:
        """

        nx_graph = self.nx_connected_graph
        raw_img = self.img

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
        nodes = nx_graph.nodes()
        gn = np.array([nodes[i]['o'] for i in nodes])
        ax.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)

        return fig, GraphStruct.plot_to_img(fig)

    def create_filenames(self, image_path):
        """
            Making the new filenames
        :return:
        """
        _, filename = os.path.split(image_path)
        output_location = self.output_path

        filename = re.sub('.png', '', filename)
        filename = re.sub('.tif', '', filename)
        filename = re.sub('.jpg', '', filename)
        filename = re.sub('.jpeg', '', filename)

        return filename, output_location

    def save_files(self, opt_gte):
        """

        :param opt_gte:
        :return:
        """

        nx_graph = self.nx_graph
        filename, output_location = self.create_filenames(self.img_path)
        g_filename = filename + "_graph.gexf"
        el_filename = filename + "_EL.csv"
        pr_filename = filename + "_processed.jpg"
        bin_filename = filename + "_binary.jpg"
        net_filename = filename + "_final.jpg"
        gexf_file = os.path.join(output_location, g_filename)
        csv_file = os.path.join(output_location, el_filename)
        img_file = os.path.join(output_location, pr_filename)
        bin_file = os.path.join(output_location, bin_filename)
        net_file = os.path.join(output_location, net_filename)

        cv2.imwrite(img_file, self.img_filtered)
        cv2.imwrite(bin_file, self.img_bin)
        # self.img_net.save(net_file)
        # cv2.imwrite(net_file, self.img_net)

        if opt_gte.export_edge_list == 1:
            if opt_gte.weighted_by_diameter == 1:
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

    @staticmethod
    def _task_init_weight(nx_graph, s, e):
        nx_graph[s][e]['length'] = nx_graph[s][e]['weight']
        if nx_graph[s][e]['weight'] == 0:
            nx_graph[s][e]['length'] = 2
        return nx_graph

    @staticmethod
    def _task_assign_weight(configs, graph_skel, nx_graph, s, e):
        if configs.weighted_by_diameter == 1:
            ge = nx_graph[s][e]['pts']
            pix_width, wt = graph_skel.assign_weights_by_width(ge)
            nx_graph[s][e]['pixel width'] = pix_width
            nx_graph[s][e]['weight'] = wt
        else:
            del nx_graph[s][e]['weight']
        return nx_graph

    @staticmethod
    def graph_components(graph):
        """

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

    @staticmethod
    def load_img_from_file(file):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        return img

    @staticmethod
    def control_brightness(img, brightness=0, contrast=0):
        """

        :param img:
        :param brightness:
        :param contrast:
        :return:
        """
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                max_val = 255
            else:
                shadow = 0
                max_val = 255 + brightness
            alpha_b = (max_val - shadow) / 255
            gamma_b = shadow
            new_img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
        else:
            new_img = img

        if contrast != 0:
            alpha_c = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            gamma_c = 127 * (1 - alpha_c)
            new_img = cv2.addWeighted(new_img, alpha_c, new_img, 0, gamma_c)

        # text string in the image.
        # cv2.putText(new_img, 'B:{},C:{}'.format(brightness, contrast), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        # 1, (0, 0, 255), 2)
        return new_img

    @staticmethod
    def plot_to_img(fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        if fig:
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
            return img
