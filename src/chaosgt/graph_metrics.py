# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Compute graph theory metrics
"""

import cv2
import csv
import math
import time
import datetime
import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
from time import sleep
from statistics import stdev, StatisticsError
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from sklearn.cluster import spectral_clustering
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality, eigenvector_centrality
from networkx.algorithms import average_node_connectivity, global_efficiency, clustering, average_clustering
from networkx.algorithms import degree_assortativity_coefficient
from networkx.algorithms.flow import maximum_flow
from networkx.algorithms.distance_measures import diameter, periphery
from networkx.algorithms.wiener import wiener_index


class GraphMetrics:

    def __init__(self, g_obj, configs):
        self.g_struct = g_obj
        self.configs = configs
        self.output_data = pd.DataFrame([])
        self.degree_distribution = [0]
        self.clustering_coefficients = [0]
        self.betweenness_distribution = [0]
        self.closeness_distribution = [0]
        self.eigenvector_distribution = [0]
        self.nx_subgraph_components = []
        self.weighted_output_data = pd.DataFrame([])
        self.weighted_degree_distribution = [0]
        self.weighted_clustering_coefficients = [0]  # NOT USED
        self.weighted_betweenness_distribution = [0]
        self.weighted_closeness_distribution = [0]
        self.weighted_eigenvector_distribution = [0]

    def compute_gt_metrics(self):
        """

        :return:
        """

        graph = self.g_struct.nx_graph
        options = self.configs
        data_dict = {"x": [], "y": []}

        node_count = int(nx.number_of_nodes(graph))
        edge_count = int(nx.number_of_edges(graph))

        data_dict["x"].append("Number of nodes")
        data_dict["y"].append(node_count)

        data_dict["x"].append("Number of edges")
        data_dict["y"].append(edge_count)

        # settings.progress(35)
        # calculating parameters as requested

        # creating degree histogram
        if options.display_degree_histogram == 1:
            print("Computing graph degree...")
            # settings.update_label("Calculating degree...")
            deg_distribution_1 = nx.degree(graph)
            deg_sum = 0
            deg_distribution = np.zeros(len(deg_distribution_1))
            for j in range(len(deg_distribution_1)):
                deg_sum += deg_distribution_1[j]
                deg_distribution[j] = deg_distribution_1[j]
            deg = deg_sum / len(deg_distribution_1)
            deg = round(deg, 5)
            self.degree_distribution = deg_distribution
            data_dict["x"].append("Average degree")
            data_dict["y"].append(deg)

        # settings.progress(40)
        if (options.compute_network_diameter == 1) or (options.compute_nodal_connectivity == 1):
            connected_graph = nx.is_connected(graph)
        else:
            connected_graph = None

        # calculating network diameter
        if options.compute_network_diameter == 1:
            print("Computing network diameter...")
            # settings.update_label("Calculating diameter...")
            if connected_graph:
                dia = int(diameter(graph))
            else:
                dia = 'NaN'
            data_dict["x"].append("Network diameter")
            data_dict["y"].append(dia)

        # calculating average nodal connectivity
        if options.compute_nodal_connectivity == 1:
            print("Computing nodal connectivity...")
            # settings.update_label("Calculating connectivity...")
            if connected_graph:
                avg_nodal_con = average_node_connectivity(graph)
                avg_nodal_con = round(avg_nodal_con, 5)
            else:
                avg_nodal_con = 'NaN'
            data_dict["x"].append("Average nodal connectivity")
            data_dict["y"].append(avg_nodal_con)

        # settings.progress(45)
        # calculating graph density
        if options.compute_graph_density == 1:
            print("Computing graph density...")
            # settings.update_label("Calculating density...")
            g_density = nx.density(graph)
            g_density = round(g_density, 5)
            data_dict["x"].append("Graph density")
            data_dict["y"].append(g_density)

        # settings.progress(50)
        # calculating global efficiency
        if options.compute_global_efficiency == 1:
            print("Computing global efficiency...")
            # settings.update_label("Calculating efficiency...")
            g_eff = global_efficiency(graph)
            g_eff = round(g_eff, 5)
            data_dict["x"].append("Global efficiency")
            data_dict["y"].append(g_eff)

        if options.compute_wiener_index == 1:
            print("Computing wiener index...")
            # settings.update_label("Calculating w_index...")
            w_index = wiener_index(graph)
            w_index = round(w_index, 1)
            data_dict["x"].append("Wiener Index")
            data_dict["y"].append(w_index)

        # settings.progress(55)
        # calculating assortativity coefficient
        if options.compute_assortativity_coef == 1:
            print("Computing assortativity coefficient...")
            # settings.update_label("Calculating assortativity...")
            a_coef = degree_assortativity_coefficient(graph)
            a_coef = round(a_coef, 5)
            data_dict["x"].append("Assortativity coefficient")
            data_dict["y"].append(a_coef)

        # settings.progress(60)
        # calculating clustering coefficients
        if (options.is_multigraph == 0) and (options.compute_clustering_coef == 1):
            print("Computing clustering coefficients...")
            # settings.update_label("Calculating clustering...")
            sleep(5)
            avg_coefficients_1 = clustering(graph)
            avg_coefficients = np.zeros(len(avg_coefficients_1))
            for j in range(len(avg_coefficients_1)):
                avg_coefficients[j] = avg_coefficients_1[j]
            clust = average_clustering(graph)
            clust = round(clust, 5)
            self.clustering_coefficients = avg_coefficients
            data_dict["x"].append("Average clustering coefficient")
            data_dict["y"].append(clust)

        # settings.progress(65)
        # calculating betweenness centrality histogram
        if (options.is_multigraph == 0) and (options.display_betweenness_histogram == 1):
            print("Computing betweenness centrality...")
            # settings.update_label("Calculating betweenness...")
            b_distribution_1 = betweenness_centrality(graph)
            b_sum = 0
            b_distribution = np.zeros(len(b_distribution_1))
            for j in range(len(b_distribution_1)):
                b_sum += b_distribution_1[j]
                b_distribution[j] = b_distribution_1[j]
            b_val = b_sum / len(b_distribution_1)
            b_val = round(b_val, 5)
            self.betweenness_distribution = b_distribution
            data_dict["x"].append("Average betweenness centrality")
            data_dict["y"].append(b_val)

        # settings.progress(70)
        # calculating eigenvector centrality
        if (options.is_multigraph == 0) and (options.display_eigenvector_histogram == 1):
            print("Computing eigenvector centrality...")
            # settings.update_label("Calculating eigenvector...")
            try:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=100)
            except nx.NetworkXPointlessConcept or nx.NetworkXError or nx.PowerIterationFailedConvergence:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=10000)
            e_sum = 0
            e_vecs = np.zeros(len(e_vecs_1))
            for j in range(len(e_vecs_1)):
                e_sum += e_vecs_1[j]
                e_vecs[j] = e_vecs_1[j]
            e_val = e_sum / len(e_vecs_1)
            e_val = round(e_val, 5)
            self.eigenvector_distribution = e_vecs
            data_dict["x"].append("Average eigenvector centrality")
            data_dict["y"].append(e_val)

        # settings.progress(75)
        # calculating closeness centrality
        if options.display_closeness_histogram == 1:
            print("Computing closeness centrality...")
            # settings.update_label("Calculating closeness...")
            close_distribution_1 = closeness_centrality(graph)
            c_sum = 0
            close_distribution = np.zeros(len(close_distribution_1))
            for j in range(len(close_distribution_1)):
                c_sum += close_distribution_1[j]
                close_distribution[j] = close_distribution_1[j]
            c_val = c_sum / len(close_distribution_1)
            c_val = round(c_val, 5)
            self.closeness_distribution = close_distribution
            data_dict["x"].append("Average closeness centrality")
            data_dict["y"].append(c_val)

        # settings.progress(80)
        # calculating graph conductance
        if options.compute_graph_conductance == 1:
            print("Computing graph conductance...")
            res_items, sg_components = self.approx_conductance_by_spectral()
            for item in res_items:
                data_dict["x"].append(item["name"])
                data_dict["y"].append(item["value"])
            self.nx_subgraph_components = sg_components
        self.output_data = pd.DataFrame(data_dict)

    def compute_weighted_gt_metrics(self):
        # settings.update_label("Performing weighted analysis...")

        graph = self.g_struct.nx_graph
        options = self.configs
        data_dict = {"x": [], "y": []}

        if options.display_degree_histogram == 1:
            deg_distribution_1 = nx.degree(graph, weight='weight')
            deg_sum = 0
            deg_distribution = np.zeros(len(deg_distribution_1))
            for j in range(len(deg_distribution_1)):
                deg_sum += deg_distribution_1[j]
                deg_distribution[j] = deg_distribution_1[j]
            deg = deg_sum / len(deg_distribution_1)
            deg = round(deg, 5)
            self.weighted_degree_distribution = deg_distribution
            data_dict["x"].append("Weighted average degree")
            data_dict["y"].append(deg)

        if options.compute_wiener_index == 1:
            w_index = wiener_index(graph, weight='length')
            w_index = round(w_index, 1)
            data_dict["x"].append("Length-weighted Wiener Index")
            data_dict["y"].append(w_index)

        if options.compute_nodal_connectivity == 1:
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
                max_flow = 'NaN'
            data_dict["x"].append("Max flow between periphery")
            data_dict["y"].append(max_flow)

        if options.compute_assortativity_coef == 1:
            a_coef = degree_assortativity_coefficient(graph, weight='pixel width')
            a_coef = round(a_coef, 5)
            data_dict["x"].append("Weighted assortativity coefficient")
            data_dict["y"].append(a_coef)

        if options.display_betweenness_histogram == 1:
            b_distribution_1 = betweenness_centrality(graph, weight='weight')
            b_sum = 0
            b_distribution = np.zeros(len(b_distribution_1))
            for j in range(len(b_distribution_1)):
                b_sum += b_distribution_1[j]
                b_distribution[j] = b_distribution_1[j]
            b_val = b_sum / len(b_distribution_1)
            b_val = round(b_val, 5)
            self.weighted_betweenness_distribution = b_distribution
            data_dict["x"].append("Width-weighted average betweenness centrality")
            data_dict["y"].append(b_val)

        if options.display_closeness_histogram == 1:
            close_distribution_1 = closeness_centrality(graph, distance='length')
            c_sum = 0
            close_distribution = np.zeros(len(close_distribution_1))
            for j in range(len(close_distribution_1)):
                c_sum += close_distribution_1[j]
                close_distribution[j] = close_distribution_1[j]
            c_val = c_sum / len(close_distribution_1)
            c_val = round(c_val, 5)
            self.weighted_closeness_distribution = close_distribution
            data_dict["x"].append("Length-weighted average closeness centrality")
            data_dict["y"].append(c_val)

        if options.display_eigenvector_histogram == 1:
            try:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=100, weight='weight')
            except nx.NetworkXPointlessConcept or nx.NetworkXError or nx.PowerIterationFailedConvergence:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=10000, weight='weight')
            e_sum = 0
            e_vecs = np.zeros(len(e_vecs_1))
            for j in range(len(e_vecs_1)):
                e_sum += e_vecs_1[j]
                e_vecs[j] = e_vecs_1[j]
            e_val = e_sum / len(e_vecs_1)
            e_val = round(e_val, 5)
            self.weighted_eigenvector_distribution = e_vecs
            data_dict["x"].append("Width-weighted average eigenvector centrality")
            data_dict["y"].append(e_val)

        self.weighted_output_data = pd.DataFrame(data_dict)

    def generate_pdf_output(self):
        """

        :return:
        """

        opt_img = self.g_struct.configs_img
        opt_gte = self.g_struct.configs_graph
        opt_gtc = self.configs

        deg_distribution = self.degree_distribution
        w_deg_distribution = self.weighted_degree_distribution
        cluster_coefs = self.clustering_coefficients
        # w_cluster_coefs = self.weighted_clustering_coefficients
        bet_distribution = self.betweenness_distribution
        w_bet_distribution = self.weighted_betweenness_distribution
        clo_distribution = self.closeness_distribution
        w_clo_distribution = self.weighted_closeness_distribution
        eig_distribution = self.eigenvector_distribution
        w_eig_distribution = self.weighted_eigenvector_distribution

        data = self.output_data
        w_data = self.weighted_output_data
        nx_graph = self.g_struct.nx_graph

        raw_img = self.g_struct.img
        filtered_img = self.g_struct.img_filtered
        img_bin = self.g_struct.img_bin
        img_histogram = cv2.calcHist([filtered_img], [0], None, [256], [0, 256])
        pdf_file, gexf_file, csv_file = self.g_struct.create_filenames(self.g_struct.img_path)

        print("Generating PDF GT Output...")
        # update_label("Generating PDF GT Output...")
        with (PdfPages(pdf_file) as pdf):
            font_1 = {'fontsize': 12}
            font_2 = {'fontsize': 9}
            # plotting the original, processed, and binary image, as well as the histogram of pixel grayscale values
            f1 = plt.figure(figsize=(8.5, 8.5), dpi=400)
            f1.add_subplot(2, 2, 1)
            plt.imshow(raw_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("Original Image")
            f1.add_subplot(2, 2, 2)
            plt.imshow(filtered_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("Processed Image")
            f1.add_subplot(2, 2, 3)
            plt.plot(img_histogram)
            if opt_img.threshold_type == 0:
                thresh_arr = np.array([[opt_img.threshold_global, opt_img.threshold_global],
                                       [0, max(img_histogram)]], dtype='object')
                plt.plot(thresh_arr[0], thresh_arr[1], ls='--', color='black')
            elif opt_img.threshold_type == 2:
                thresh_arr = np.array([[self.g_struct.otsu_val, self.g_struct.otsu_val],
                                       [0, max(img_histogram)]], dtype='object')
                plt.plot(thresh_arr[0], thresh_arr[1], ls='--', color='black')
            plt.yticks([])
            plt.title("Histogram of Processed Image")
            plt.xlabel("Pixel values")
            plt.ylabel("Counts")
            f1.add_subplot(2, 2, 4)
            plt.imshow(img_bin, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("Binary Image")

            pdf.savefig()
            plt.close()

            # plotting skeletal images
            g_skel = self.g_struct.graph_skeleton
            f2a = plt.figure(figsize=(8.5, 11), dpi=400)
            f2a.add_subplot(2, 1, 1)
            # g_skel.skel_int = -1*(g_skel.skel_int-1)
            plt.imshow(g_skel.skel_int, cmap='gray')
            plt.scatter(g_skel.bp_coord_x, g_skel.bp_coord_y, s=0.25, c='b')
            plt.scatter(g_skel.ep_coord_x, g_skel.ep_coord_y, s=0.25, c='r')
            plt.xticks([])
            plt.yticks([])
            plt.title("Skeletal Image")
            f2a.add_subplot(2, 1, 2)
            plt.imshow(raw_img, cmap='gray')
            if opt_gte.is_multigraph:
                for (s, e) in nx_graph.edges():
                    for k in range(int(len(nx_graph[s][e]))):
                        ge = nx_graph[s][e][k]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'red')
            else:
                for (s, e) in nx_graph.edges():
                    ge = nx_graph[s][e]['pts']
                    plt.plot(ge[:, 1], ge[:, 0], 'red')

            # plotting the final graph with the nodes
            nodes = nx_graph.nodes()
            gn = np.array([nodes[i]['o'] for i in nodes])
            if opt_gte.display_node_id == 1:
                i = 0
                for x, y in zip(gn[:, 1], gn[:, 0]):
                    plt.annotate(str(i), (x, y), fontsize=5)
                    i += 1
                plt.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)

            else:
                plt.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)
            plt.xticks([])
            plt.yticks([])
            plt.title("Final Graph")
            pdf.savefig()
            plt.close()

            # plotting sub-graph network
            f2b = plt.figure(figsize=(8.5, 11), dpi=400)
            f2b.add_subplot(1, 1, 1)
            plt.imshow(raw_img, cmap='gray')
            color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            color_cycle = cycle(color_list)
            for component in self.nx_subgraph_components:
                sg = nx_graph.subgraph(component)
                color = next(color_cycle)
                for (s, e) in sg.edges():
                    ge = sg[s][e]['pts']
                    plt.plot(ge[:, 1], ge[:, 0], color)
            plt.xticks([])
            plt.yticks([])
            plt.title("Sub Graphs")
            pdf.savefig()
            plt.close()

            # progress(95)
            # displaying all the GT calculations requested
            # Modified Output to show GI calculations on entire page
            if opt_gte.weighted_by_diameter == 1:
                f3a = plt.figure(figsize=(8.5, 11), dpi=300)
                f3a.add_subplot(1, 1, 1)
                f3a.patch.set_visible(False)
                plt.axis('off')
                col_width = [2 / 3, 1 / 3]
                table = plt.table(cellText=data.values[:, :], loc='upper center', colWidths=col_width, cellLoc='left')
                table.scale(1, 1.5)
                plt.title("Unweighted GT parameters")
                pdf.savefig()
                plt.close()
                # try:  # generates exception is w_data is None
                f3b = plt.figure(figsize=(8.5, 11), dpi=300)
                f3b.add_subplot(1, 1, 1)
                f3b.patch.set_visible(False)
                plt.axis('off')
                col_width = [2 / 3, 1 / 3]
                tab_2 = plt.table(cellText=w_data.values[:, :], loc='upper center', colWidths=col_width, cellLoc='left')
                tab_2.scale(1, 1.5)
                plt.title("Weighted GT Parameters")
                pdf.savefig()
                plt.close()
                # except:
                #    pass
            else:
                f3a = plt.figure(figsize=(8.5, 11), dpi=300)
                f3a.add_subplot(1, 1, 1)
                f3a.patch.set_visible(False)
                plt.axis('off')
                col_width = [2 / 3, 1 / 3]
                table = plt.table(cellText=data.values[:, :], loc='upper center', colWidths=col_width, cellLoc='left')
                table.scale(1, 1.5)
                plt.title("Unweighted GT parameters")
                pdf.savefig()
                plt.close()

                f3b = plt.figure(figsize=(8.5, 11), dpi=300)
                if opt_gtc.display_degree_histogram == 1:
                    f3b.add_subplot(2, 2, 1)
                    bins_1 = np.arange(0.5, max(deg_distribution) + 1.5, 1)
                    try:
                        deg_val = str(round(stdev(deg_distribution), 3))
                    except StatisticsError:
                        deg_val = "N/A"
                    deg_txt = r'Degree Distribution: $\sigma$=' + str(deg_val)
                    plt.hist(deg_distribution, bins=bins_1)
                    plt.title(deg_txt)
                    plt.xlabel("Degree")
                    plt.ylabel("Counts")
                if (opt_gtc.compute_clustering_coef == 1) and (opt_gte.is_multigraph == 0):
                    f3b.add_subplot(2, 2, 2)
                    bins_t = np.linspace(min(cluster_coefs), max(cluster_coefs), 50)
                    try:
                        t_val = str(round(stdev(cluster_coefs), 3))
                    except StatisticsError:
                        t_val = "N/A"
                    t_txt = r"Clustering Coefficients: $\sigma$=" + str(t_val)
                    plt.hist(cluster_coefs, bins=bins_t)
                    plt.title(t_txt)
                    plt.xlabel("Clust. Coeff.")
                    plt.ylabel("Counts")
                pdf.savefig()
                plt.close()

            if (opt_gte.is_multigraph == 0) and (opt_gte.weighted_by_diameter == 0):
                if (opt_gtc.display_betweenness_histogram == 1) or (opt_gtc.display_closeness_histogram == 1) or\
                        (opt_gtc.display_eigenvector_histogram == 1):
                    f4 = plt.figure(figsize=(8.5, 11), dpi=400)
                    if opt_gtc.display_betweenness_histogram == 1:
                        f4.add_subplot(2, 2, 1)
                        bins_2 = np.linspace(min(bet_distribution), max(bet_distribution), 50)
                        try:
                            bt_val = str(round(stdev(bet_distribution), 3))
                        except StatisticsError:
                            bt_val = "N/A"
                        bc_txt = r"Betweenness Centrality: $\sigma$=" + str(bt_val)
                        plt.hist(bet_distribution, bins=bins_2)
                        plt.title(bc_txt)
                        plt.xlabel("Betweenness value")
                        plt.ylabel("Counts")
                    if opt_gtc.display_closeness_histogram == 1:
                        f4.add_subplot(2, 2, 2)
                        bins_3 = np.linspace(min(clo_distribution), max(clo_distribution), 50)
                        try:
                            cc_val = str(round(stdev(clo_distribution), 3))
                        except StatisticsError:
                            cc_val = "N/A"
                        cc_txt = r"Closeness Centrality: $\sigma$=" + str(cc_val)
                        plt.hist(clo_distribution, bins=bins_3)
                        plt.title(cc_txt)
                        plt.xlabel("Closeness value")
                        plt.ylabel("Counts")
                    if opt_gtc.display_eigenvector_histogram == 1:
                        f4.add_subplot(2, 2, 3)
                        bins4 = np.linspace(min(eig_distribution), max(eig_distribution), 50)
                        try:
                            ec_val = str(round(stdev(eig_distribution), 3))
                        except StatisticsError:
                            ec_val = "N/A"
                        ec_txt = r"Eigenvector Centrality: $\sigma$=" + str(ec_val)
                        plt.hist(eig_distribution, bins=bins4)
                        plt.title(ec_txt)
                        plt.xlabel("Eigenvector value")
                        plt.ylabel("Counts")
                    pdf.savefig()
                    plt.close()

            # displaying weighted GT parameters if requested
            if opt_gte.weighted_by_diameter == 1:
                if opt_gte.is_multigraph:
                    g_count_1 = opt_gtc.display_degree_histogram + opt_gtc.display_closeness_histogram
                else:
                    g_count_1 = opt_gtc.display_degree_histogram + opt_gtc.compute_clustering_coef +\
                                opt_gtc.display_betweenness_histogram + opt_gtc.display_closeness_histogram +\
                                opt_gtc.display_eigenvector_histogram
                g_count_2 = g_count_1 - opt_gtc.compute_clustering_coef + 1
                index = 1
                if g_count_1 > 2:
                    sy_1 = 2
                    fnt = font_2
                else:
                    sy_1 = 1
                    fnt = font_1
                f4 = plt.figure(figsize=(8.5, 11), dpi=400)
                if opt_gtc.display_degree_histogram:
                    f4.add_subplot(sy_1, 2, index)
                    bins_1 = np.arange(0.5, max(deg_distribution) + 1.5, 1)
                    try:
                        deg_val = str(round(stdev(deg_distribution), 3))
                    except StatisticsError:
                        deg_val = "N/A"
                    deg_txt = r"Degree Distribution: $\sigma$=" + str(deg_val)
                    plt.hist(deg_distribution, bins=bins_1)
                    plt.title(deg_txt, fontdict=fnt)
                    plt.xlabel("Degree", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.is_multigraph == 0):
                    f4.add_subplot(sy_1, 2, index)
                    bins_2 = np.linspace(min(bet_distribution), max(bet_distribution), 50)
                    try:
                        bt_val = str(round(stdev(bet_distribution), 3))
                    except StatisticsError:
                        bt_val = "N/A"
                    bc_txt = r"Betweenness Centrality: $\sigma$=" + str(bt_val)
                    plt.hist(bet_distribution, bins=bins_2)
                    plt.title(bc_txt, fontdict=fnt)
                    plt.xlabel("Betweenness value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if opt_gtc.display_closeness_histogram == 1:
                    f4.add_subplot(sy_1, 2, index)
                    bins_3 = np.linspace(min(clo_distribution), max(clo_distribution), 50)
                    try:
                        cc_val = str(round(stdev(clo_distribution), 3))
                    except StatisticsError:
                        cc_val = "N/A"
                    cc_txt = r"Closeness Centrality: $\sigma$=" + str(cc_val)
                    plt.hist(clo_distribution, bins=bins_3)
                    plt.title(cc_txt, fontdict=fnt)
                    plt.xlabel("Closeness value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.is_multigraph == 0):
                    f4.add_subplot(sy_1, 2, index)
                    bins4 = np.linspace(min(eig_distribution), max(eig_distribution), 50)
                    try:
                        ec_val = str(round(stdev(eig_distribution), 3))
                    except StatisticsError:
                        ec_val = "N/A"
                    bc_txt = r"Eigenvector Centrality: $\sigma$=" + str(ec_val)
                    plt.hist(bet_distribution, bins=bins4)
                    plt.title(bc_txt, fontdict=fnt)
                    plt.xlabel("Eigenvector value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                pdf.savefig()
                plt.close()

                f5 = plt.figure(figsize=(8.5, 11), dpi=400)
                if g_count_2 > 2:
                    sy_2 = 2
                    fnt = font_2
                else:
                    sy_2 = 1
                    fnt = font_1
                index = 1
                if opt_gtc.display_degree_histogram:
                    f5.add_subplot(sy_2, 2, index)
                    bins4 = np.arange(0.5, max(w_deg_distribution) + 1.5, 1)
                    try:
                        w_deg_val = str(round(stdev(w_deg_distribution), 3))
                    except StatisticsError:
                        w_deg_val = "N/A"
                    w_deg_txt = r"Weighted Degree: $\sigma$=" + str(w_deg_val)
                    plt.hist(w_deg_distribution, bins=bins4)
                    plt.title(w_deg_txt, fontdict=fnt)
                    plt.xlabel("Degree", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.is_multigraph == 0):
                    f5.add_subplot(sy_2, 2, index)
                    bins_5 = np.linspace(min(w_bet_distribution), max(w_bet_distribution), 50)
                    plt.hist(w_bet_distribution, bins=bins_5)
                    try:
                        w_bt_val = str(round(stdev(w_bet_distribution), 3))
                    except StatisticsError:
                        w_bt_val = "N/A"
                    w_bt_txt = r"Width-Weighted Betweenness: $\sigma$=" + str(w_bt_val)
                    plt.title(w_bt_txt, fontdict=fnt)
                    plt.xlabel("Betweenness value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if opt_gtc.display_closeness_histogram == 1:
                    f5.add_subplot(sy_2, 2, index)
                    bins_6 = np.linspace(min(w_clo_distribution), max(w_clo_distribution), 50)
                    try:
                        w_clo_val = str(round(stdev(w_clo_distribution), 3))
                    except StatisticsError:
                        w_clo_val = "N/A"
                    w_clo_txt = r"Length-Weighted Closeness: $\sigma$=" + str(w_clo_val)
                    plt.hist(w_clo_distribution, bins=bins_6)
                    plt.title(w_clo_txt, fontdict=fnt)
                    plt.xlabel("Closeness value", fontdict=fnt)
                    plt.xticks(fontsize=8)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.is_multigraph == 0):
                    f5.add_subplot(sy_2, 2, index)
                    bins7 = np.linspace(min(w_eig_distribution), max(w_eig_distribution), 50)
                    plt.hist(w_eig_distribution, bins=bins7)
                    try:
                        w_ec_val = str(round(stdev(w_eig_distribution), 3))
                    except StatisticsError:
                        w_ec_val = "N/A"
                    w_ec_txt = r"Width-Weighted Eigenvector Cent.: $\sigma$=" + str(w_ec_val)
                    plt.title(w_ec_txt, fontdict=fnt)
                    plt.xlabel("Eigenvector value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                pdf.savefig()
                plt.close()

            # displaying heatmaps
            if opt_gtc.display_heatmaps:
                sz = 30
                lw = 1.5
                # update_label("Generating heat maps...")
                time.sleep(0.5)
                if opt_gtc.display_degree_histogram == 1:
                    f6a = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6a.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=deg_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Degree Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_degree_histogram == 1) and (opt_gte.weighted_by_diameter == 1):
                    f6b = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6b.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_deg_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Weighted Degree Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.compute_clustering_coef == 1) and (opt_gte.is_multigraph == 0):
                    f6c = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6c.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=cluster_coefs, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Clustering Coefficient Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.is_multigraph == 0):
                    f6d = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6d.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=bet_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Betweenness Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_betweenness_histogram == 1) and (opt_gte.weighted_by_diameter == 1) and\
                        (opt_gte.is_multigraph == 0):
                    f6e = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6e.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_bet_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Width-Weighted Betweenness Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if opt_gtc.display_closeness_histogram == 1:
                    f6f = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6f.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=clo_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Closeness Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_closeness_histogram == 1) and (opt_gte.weighted_by_diameter == 1):
                    f6f = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6f.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_clo_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Length-Weighted Closeness Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.is_multigraph == 0):
                    f6h = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6h.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=eig_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Eigenvector Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (opt_gtc.display_eigenvector_histogram == 1) and (opt_gte.weighted_by_diameter == 1) and\
                        (opt_gte.is_multigraph == 0):
                    f6h = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6h.add_subplot(1, 1, 1)
                    plt.imshow(raw_img, cmap='gray')
                    nodes = nx_graph.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_eig_distribution, cmap='plasma')
                    if opt_gte.is_multigraph:
                        for (s, e) in nx_graph.edges():
                            for k in range(int(len(nx_graph[s][e]))):
                                ge = nx_graph[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in nx_graph.edges():
                            ge = nx_graph[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Width-Weighted Eigenvector Centrality Heatmap', fontdict=font_1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()

            # displaying run information
            f8 = plt.figure(figsize=(8.5, 8.5), dpi=300)
            f8.add_subplot(1, 1, 1)
            plt.text(0.5, 0.5, self.get_info(), horizontalalignment='center', verticalalignment='center')
            plt.xticks([])
            plt.yticks([])
            pdf.savefig()
            plt.close()

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

        # exporting as gephi file
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

    def get_info(self):
        # similar to the start of the csv file, this is just getting all the relevant settings to display in the pdf
        opt_img = self.g_struct.configs_img
        opt_gte = self.g_struct.configs_graph
        run_info = "Run Info\n"
        run_info = run_info + self.g_struct.img_path
        now = datetime.datetime.now()
        run_info = run_info + " || " + now.strftime("%Y-%m-%d %H:%M:%S") + "\n"
        if opt_img.threshold_type == 0:
            run_info = run_info + " || Global Threshold (" + str(opt_img.threshold_global) + ")"
        elif opt_img.threshold_type == 1:
            run_info = run_info + " || Adaptive Threshold, " + str(opt_img.threshold_adaptive) + " bit kernel"
        elif opt_img.threshold_type == 2:
            run_info = run_info + " || OTSU Threshold"
        if opt_img.gamma != 1:
            run_info = run_info + "|| Gamma = " + str(opt_img.gamma)
        if opt_img.apply_median:
            run_info = run_info + " || Median Filter"
        if opt_img.apply_gaussian:
            run_info = run_info + " || Gaussian Blur, " + str(opt_img.blurring_window_size) + " bit kernel"
        if opt_img.apply_autolevel:
            run_info = run_info + " || Autolevel"
        if opt_img.apply_dark_foreground:
            run_info = run_info + " || Dark Foreground"
        if opt_img.apply_laplacian:
            run_info = run_info + " || Laplacian Gradient"
        if opt_img.apply_scharr:
            run_info = run_info + " || Scharr Gradient"
        if opt_img.apply_sobel:
            run_info = run_info + " || Sobel Gradient"
        if opt_img.apply_lowpass:
            run_info = run_info + " || Low-pass filter" + str(opt_img.filter_window_size)
        run_info = run_info + "\n"
        if opt_gte.merge_nearby_nodes:
            run_info = run_info + " || Merge Nodes"
        if opt_gte.prune_dangling_edges:
            run_info = run_info + " || Prune Dangling Edges"
        if opt_gte.remove_disconnected_segments:
            run_info = run_info + " || Remove Objects of Size " + str(opt_gte.remove_object_size)
        if opt_gte.remove_self_loops:
            run_info = run_info + " || Remove Self Loops"
        if opt_gte.is_multigraph:
            run_info = run_info + " || Multi-graph allowed"
        return run_info

    def approx_conductance_by_spectral(self):
        """
            https://doi.org/10.1016/j.procs.2013.09.311

            Conductance can closely be approximated via eigenvalue computation,\
        a fact which has been well-known and well-used in the graph theory community.\

            The Laplacian matrix of a directed graph is by definition generally non-symmetric,\
        while, e.g., traditional spectral clustering is primarily developed for undirected\
        graphs with symmetric adjacency and Laplacian matrices. A trivial approach to apply\
        techniques requiring the symmetry is to turn the original directed graph into an\
        undirected graph and build the Laplacian matrix for the latter.\

            We need to remove isolated nodes (in order to avoid singular adjacency matrix).\
        The degree of a node is the number of edges incident to that node.\
        When a node has a degree of zero, it means that there are no edges\
        connected to that node. In other words, the node is isolated from\
        the rest of the graph.

        """

        # It is important to notice our graph is (mostly) a directed graph,
        # meaning that it is: (asymmetric) with self-looping nodes

        graph = self.g_struct.nx_graph
        data = []
        sub_components = []

        # 1. Make a copy of the graph
        # eig_graph = graph.copy()

        # 2a. Remove self-looping edges
        eig_graph = GraphMetrics.remove_self_loops(graph)

        # 2b. Identify isolated nodes
        isolated_nodes = list(nx.isolates(eig_graph))

        # 2c. Remove isolated nodes
        eig_graph.remove_nodes_from(isolated_nodes)

        # 3a. Check connectivity of graph
        try:
            # does not work if graphs has disconnected sub-graphs
            nx.fiedler_vector(eig_graph)
        except nx.NetworkXNotImplemented:
            # Graph is directed.
            non_directed_graph = GraphMetrics.make_graph_symmetrical(eig_graph)
            try:
                nx.fiedler_vector(non_directed_graph)
            except nx.NetworkXNotImplemented:
                print("Graph is directed. Cannot compute conductance")
                return None
            except nx.NetworkXError:
                # Graph has less than two nodes or is not connected.
                sub_graph_largest, sub_graph_smallest, size, sub_components = GraphMetrics.graph_components(eig_graph)
                eig_graph = sub_graph_largest
                data.append({"name": "Subgraph Count", "value": size})
                data.append({"name": "Large Subgraph Node Count", "value": sub_graph_largest.number_of_nodes()})
                data.append({"name": "Large Subgraph Edge Count", "value": sub_graph_largest.number_of_edges()})
                data.append({"name": "Small Subgraph Node Count", "value": sub_graph_smallest.number_of_nodes()})
                data.append({"name": "Small Subgraph Edge Count", "value": sub_graph_smallest.number_of_edges()})
        except nx.NetworkXError:
            # Graph has less than two nodes or is not connected.
            sub_graph_largest, sub_graph_smallest, size, sub_components = GraphMetrics.graph_components(eig_graph)
            eig_graph = sub_graph_largest
            data.append({"name": "Subgraph Count", "value": size})
            data.append({"name": "Large Subgraph Node Count", "value": sub_graph_largest.number_of_nodes()})
            data.append({"name": "Large Subgraph Edge Count", "value": sub_graph_largest.number_of_edges()})
            data.append({"name": "Small Subgraph Node Count", "value": sub_graph_smallest.number_of_nodes()})
            data.append({"name": "Small Subgraph Edge Count", "value": sub_graph_smallest.number_of_edges()})

        # 4. Compute normalized-laplacian matrix
        # norm_laplacian_matrix = compute_norm_laplacian_matrix(eig_graph)
        norm_laplacian_matrix = nx.normalized_laplacian_matrix(eig_graph).toarray()

        # 5. Compute eigenvalues
        # e_vals, _ = np.linalg.eig(norm_laplacian_matrix)
        e_vals = sp.linalg.eigvals(norm_laplacian_matrix)

        # 6. Approximate conductance using the 2nd smallest eigenvalue
        eigenvalues = e_vals.real
        val_max, val_min = GraphMetrics.compute_conductance_range(eigenvalues)
        data.append({"name": "Graph Conductance (max)", "value": val_max})
        data.append({"name": "Graph Conductance (min)", "value": val_min})

        # print(val_max)
        # print(val_min)
        return data, sub_components

    @staticmethod
    def compute_norm_laplacian_matrix(graph):
        """
        Compute normalized-laplacian-matrix

        :param graph:
        :return:
        """

        # 1. Get Adjacency matrix
        adj_mat = nx.adjacency_matrix(graph).todense()

        # 2. Compute Degree matrix
        deg_mat = np.diag(np.sum(adj_mat, axis=1))

        # 3. Compute Identity matrix
        id_mat = np.identity(adj_mat.shape[0])

        # 4. Compute (Degree inverse squared) D^{-1/2} matrix
        # Check for singular matrices
        if np.any(np.diag(deg_mat) == 0):
            # Graph has nodes with zero degree. Cannot compute inverse square root of degree matrix.
            # raise ValueError("Graph has nodes with zero degree. Cannot compute inverse square root of degree matrix.")
            print("Graph has nodes with zero degree. Cannot compute conductance")
            return None
        deg_inv_sqrt = np.linalg.inv(np.sqrt(deg_mat))

        # 5. Compute Laplacian matrix
        # lpl_mat = deg_mat - adj_mat

        # 6. Compute normalized-Laplacian matrix
        norm_lpl_mat = id_mat - np.dot(deg_inv_sqrt, np.dot(adj_mat, deg_inv_sqrt))
        # norm_lpl_mat = np.eye(sp_graph.number_of_nodes()) - np.dot(np.dot(deg_inv_sqrt, adj_mat), deg_inv_sqrt)

        # print(adj_mat)
        # print(adj_mat.shape)
        # print(deg_mat)
        # print(deg_inv_sqrt)
        # print(id_mat)
        # print(lpl_mat)
        # print(norm_lpl_mat)
        return norm_lpl_mat

    @staticmethod
    def remove_self_loops(graph):
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
    def make_graph_symmetrical(graph):
        """

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
    def compute_conductance_range(eig_vals):
        """

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
