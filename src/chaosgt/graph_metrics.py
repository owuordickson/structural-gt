# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Compute graph theory metrics
"""

import math
import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
from time import sleep
from sklearn.cluster import spectral_clustering
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
        self.output_data = None
        self.degree_distribution = [0]
        self.clustering_coefficients = [0]
        self.betweenness_distribution = [0]
        self.closeness_distribution = [0]
        self.eigenvector_distribution = [0]
        self.nx_subgraph_components = []
        self.weighted_output_data = None  # w_data
        self.weighted_degree_distribution = [0]  # w_klist
        self.weighted_clustering_coefficients = [0]  # w_Tlist
        self.weighted_betweenness_distribution = [0]  # w_BCdist
        self.weighted_closeness_distribution = [0]  # w_CCdist
        self.weighted_eigenvector_distribution = [0]  # w_ECdist

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
        # calculating network diameter
        if options.compute_network_diameter == 1:
            # settings.update_label("Calculating diameter...")
            connected_graph = nx.is_connected(graph)
            if connected_graph:
                dia = int(diameter(graph))
            else:
                dia = 'NaN'
            data_dict["x"].append("Network diameter")
            data_dict["y"].append(dia)

        # calculating average nodal connectivity
        if options.compute_nodal_connectivity == 1:
            # settings.update_label("Calculating connectivity...")
            connected_graph = nx.is_connected(graph)
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
            # settings.update_label("Calculating density...")
            g_density = nx.density(graph)
            g_density = round(g_density, 5)
            data_dict["x"].append("Graph density")
            data_dict["y"].append(g_density)

        # settings.progress(50)
        # calculating global efficiency
        if options.compute_global_efficiency == 1:
            # settings.update_label("Calculating efficiency...")
            g_eff = global_efficiency(graph)
            g_eff = round(g_eff, 5)
            data_dict["x"].append("Global efficiency")
            data_dict["y"].append(g_eff)

        if options.compute_wiener_index == 1:
            # settings.update_label("Calculating w_index...")
            w_index = wiener_index(graph)
            w_index = round(w_index, 1)
            data_dict["x"].append("Wiener Index")
            data_dict["y"].append(w_index)

        # settings.progress(55)
        # calculating assortativity coefficient
        if options.compute_assortativity_coef == 1:
            # settings.update_label("Calculating assortativity...")
            a_coef = degree_assortativity_coefficient(graph)
            a_coef = round(a_coef, 5)
            data_dict["x"].append("Assortativity coefficient")
            data_dict["y"].append(a_coef)

        # settings.progress(60)
        # calculating clustering coefficients
        if (not options.disable_multigraph) and (options.compute_clustering_coef == 1):
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
        if (not options.disable_multigraph) and (options.display_betweeness_histogram == 1):
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
        if (not options.disable_multigraph) and (options.display_eigenvector_histogram == 1):
            # settings.update_label("Calculating eigenvector...")
            try:
                e_vecs_1 = eigenvector_centrality(graph, max_iter=100)
            except:
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
            res_items, sg_components = self.approx_conductance_by_spectral()
            for item in res_items:
                data_dict["x"].append(item["name"])
                data_dict["y"].append(item["value"])
        self.nx_subgraph_components = sg_components
        self.output_data = pd.DataFrame(data_dict)

    def compute_weighted_gt_metrics(self):
        pass

    def generate_pdf_output(self, data, w_data):
        # raw_img = src
        # img_filt = img
        # img_bin = img_bin
        # histo = cv2.calcHist([img_filt], [0], None, [256], [0, 256])

        # update_label("Generating PDF GT Output...")
        """"
        with PdfPages(file) as pdf:
            font1 = {'fontsize': 12}
            font2 = {'fontsize': 9}
            # plotting the original, processed, and binary image, as well as the histogram of pixel grayscale values
            f1 = plt.figure(figsize=(8.5, 8.5), dpi=400)
            f1.add_subplot(2, 2, 1)
            plt.imshow(raw_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("Original Image")
            f1.add_subplot(2, 2, 2)
            plt.imshow(img_filt, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("Processed Image")
            f1.add_subplot(2, 2, 3)
            plt.plot(histo)
            if (Thresh_method == 0):
                Th = np.array([[thresh, thresh], [0, max(histo)]], dtype='object')
                plt.plot(Th[0], Th[1], ls='--', color='black')
            elif (Thresh_method == 2):
                Th = np.array([[ret, ret], [0, max(histo)]], dtype='object')
                plt.plot(Th[0], Th[1], ls='--', color='black')
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
            f2a = plt.figure(figsize=(8.5, 11), dpi=400)
            f2a.add_subplot(2, 1, 1)
            # skel_int = -1*(skel_int-1)
            plt.imshow(skel_int, cmap='gray')
            plt.scatter(Bp_coord_x, Bp_coord_y, s=0.25, c='b')
            plt.scatter(Ep_coord_x, Ep_coord_y, s=0.25, c='r')
            plt.xticks([])
            plt.yticks([])
            plt.title("Skeletal Image")
            f2a.add_subplot(2, 1, 2)
            plt.imshow(src, cmap='gray')
            if multigraph:
                for (s, e) in G.edges():
                    for k in range(int(len(G[s][e]))):
                        ge = G[s][e][k]['pts']
                        plt.plot(ge[:, 1], ge[:, 0], 'red')
            else:
                for (s, e) in G.edges():
                    ge = G[s][e]['pts']
                    plt.plot(ge[:, 1], ge[:, 0], 'red')

            # plotting the final graph with the nodes
            nodes = G.nodes()
            gn = np.array([nodes[i]['o'] for i in nodes])
            if (display_nodeID == 1):
                i = 0;
                for x, y in zip(gn[:, 1], gn[:, 0]):
                    plt.annotate(i, (x, y), fontsize=5)
                    i += 1
                plt.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)

            else:
                plt.plot(gn[:, 1], gn[:, 0], 'b.', markersize=3)
            plt.xticks([])
            plt.yticks([])
            plt.title("Final Graph")
            pdf.savefig()
            plt.close()

            f2b = plt.figure(figsize=(8.5, 11), dpi=400)
            f2b.add_subplot(1, 1, 1)
            plt.imshow(src, cmap='gray')
            color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            color_cycle = cycle(color_list)
            for component in SGcomponents:
                sg = G.subgraph(component)
                color = next(color_cycle)
                for (s, e) in sg.edges():
                    ge = sg[s][e]['pts']
                    plt.plot(ge[:, 1], ge[:, 0], color)
            # plt.axis("off")
            plt.xticks([])
            plt.yticks([])
            plt.title("Sub Graphs")
            pdf.savefig()
            plt.close()

            progress(95)

            # displaying all the GT calculations requested
            # Modified Output to show GI calculations on entire page
            if weighted == 1:
                f3a = plt.figure(figsize=(8.5, 11), dpi=300)
                f3a.add_subplot(1, 1, 1)
                f3a.patch.set_visible(False)
                plt.axis('off')
                colw = [2 / 3, 1 / 3]
                table = plt.table(cellText=data.values[:, :], loc='upper center', colWidths=colw, cellLoc='left')
                table.scale(1, 1.5)
                plt.title("Unweighted GT parameters")
                pdf.savefig()
                plt.close()
                try:
                    f3b = plt.figure(figsize=(8.5, 11), dpi=300)
                    f3b.add_subplot(1, 1, 1)
                    f3b.patch.set_visible(False)
                    plt.axis('off')
                    colw = [2 / 3, 1 / 3]
                    table2 = plt.table(cellText=w_data.values[:, :], loc='upper center', colWidths=colw, cellLoc='left')
                    table2.scale(1, 1.5)
                    plt.title("Weighted GT Parameters")
                    pdf.savefig()
                    plt.close()
                except:
                    pass
            else:
                f3a = plt.figure(figsize=(8.5, 11), dpi=300)
                f3a.add_subplot(1, 1, 1)
                f3a.patch.set_visible(False)
                plt.axis('off')
                colw = [2 / 3, 1 / 3]
                table = plt.table(cellText=data.values[:, :], loc='upper center', colWidths=colw, cellLoc='left')
                table.scale(1, 1.5)
                plt.title("Unweighted GT parameters")
                pdf.savefig()
                plt.close()

                f3b = plt.figure(figsize=(8.5, 11), dpi=300)
                if Do_kdist:
                    f3b.add_subplot(2, 2, 1)
                    bins1 = np.arange(0.5, max(klist) + 1.5, 1)
                    try:
                        k_sig = str(round(stdev(klist), 3))
                    except:
                        k_sig = "N/A"
                    k_txt = "Degree Distribution: $\sigma$=" + k_sig
                    plt.hist(klist, bins=bins1)
                    plt.title(k_txt)
                    plt.xlabel("Degree")
                    plt.ylabel("Counts")
                if (Do_clust and multigraph == 0):
                    f3b.add_subplot(2, 2, 2)
                    binsT = np.linspace(min(Tlist), max(Tlist), 50)
                    try:
                        T_sig = str(round(stdev(Tlist), 3))
                    except:
                        T_sig = "N/A"
                    T_txt = "Clustering Coefficients: $\sigma$=" + T_sig
                    plt.hist(Tlist, bins=binsT)
                    plt.title(T_txt)
                    plt.xlabel("Clust. Coeff.")
                    plt.ylabel("Counts")
                pdf.savefig()
                plt.close()

            if (multigraph == 0 and weighted == 0):
                if (Do_BCdist or Do_CCdist or Do_ECdist):
                    f4 = plt.figure(figsize=(8.5, 11), dpi=400)
                    if Do_BCdist:
                        f4.add_subplot(2, 2, 1)
                        bins2 = np.linspace(min(BCdist), max(BCdist), 50)
                        try:
                            BC_sig = str(round(stdev(BCdist), 3))
                        except:
                            BC_sig = "N/A"
                        BC_txt = "Betweenness Centrality: $\sigma$=" + BC_sig
                        plt.hist(BCdist, bins=bins2)
                        plt.title(BC_txt)
                        plt.xlabel("Betweenness value")
                        plt.ylabel("Counts")
                    if Do_CCdist:
                        f4.add_subplot(2, 2, 2)
                        bins3 = np.linspace(min(CCdist), max(CCdist), 50)
                        try:
                            CC_sig = str(round(stdev(CCdist), 3))
                        except:
                            CC_sig = "N/A"
                        CC_txt = "Closeness Centrality: $\sigma$=" + CC_sig
                        plt.hist(CCdist, bins=bins3)
                        plt.title(CC_txt)
                        plt.xlabel("Closeness value")
                        plt.ylabel("Counts")
                    if Do_ECdist:
                        f4.add_subplot(2, 2, 3)
                        bins4 = np.linspace(min(ECdist), max(ECdist), 50)
                        try:
                            EC_sig = str(round(stdev(ECdist), 3))
                        except:
                            EC_sig = "N/A"
                        EC_txt = "Eigenvector Centrality: $\sigma$=" + EC_sig
                        plt.hist(ECdist, bins=bins4)
                        plt.title(EC_txt)
                        plt.xlabel("Eigenvector value")
                        plt.ylabel("Counts")
                try:
                    pdf.savefig()
                except:
                    None
                try:
                    plt.close()
                except:
                    None

            # displaying weighted GT parameters if requested
            if (weighted == 1):
                if multigraph:
                    Do_BCdist = 0
                    Do_ECdist = 0
                    Do_clust = 0
                g_count = Do_kdist + Do_clust + Do_BCdist + Do_CCdist + Do_ECdist
                g_count2 = g_count - Do_clust + 1
                index = 1
                if (g_count > 2):
                    sy1 = 2
                    fnt = font2
                else:
                    sy1 = 1
                    fnt = font1
                f4 = plt.figure(figsize=(8.5, 11), dpi=400)
                if Do_kdist:
                    f4.add_subplot(sy1, 2, index)
                    bins1 = np.arange(0.5, max(klist) + 1.5, 1)
                    try:
                        k_sig = str(round(stdev(klist), 3))
                    except:
                        k_sig = "N/A"
                    k_txt = "Degree Distribution: $\sigma$=" + k_sig
                    plt.hist(klist, bins=bins1)
                    plt.title(k_txt, fontdict=fnt)
                    plt.xlabel("Degree", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if Do_BCdist:
                    f4.add_subplot(sy1, 2, index)
                    bins2 = np.linspace(min(BCdist), max(BCdist), 50)
                    try:
                        BC_sig = str(round(stdev(BCdist), 3))
                    except:
                        BC_sig = "N/A"
                    BC_txt = "Betweenness Centrality: $\sigma$=" + BC_sig
                    plt.hist(BCdist, bins=bins2)
                    plt.title(BC_txt, fontdict=fnt)
                    plt.xlabel("Betweenness value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if Do_CCdist:
                    f4.add_subplot(sy1, 2, index)
                    bins3 = np.linspace(min(CCdist), max(CCdist), 50)
                    try:
                        CC_sig = str(round(stdev(CCdist), 3))
                    except:
                        CC_sig = "N/A"
                    CC_txt = "Closeness Centrality: $\sigma$=" + CC_sig
                    plt.hist(CCdist, bins=bins3)
                    plt.title(CC_txt, fontdict=fnt)
                    plt.xlabel("Closeness value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if Do_ECdist:
                    f4.add_subplot(sy1, 2, index)
                    bins4 = np.linspace(min(ECdist), max(ECdist), 50)
                    try:
                        EC_sig = str(round(stdev(ECdist), 3))
                    except:
                        EC_sig = "N/A"
                    BC_txt = "Eigenvector Centrality: $\sigma$=" + EC_sig
                    plt.hist(BCdist, bins=bins4)
                    plt.title(BC_txt, fontdict=fnt)
                    plt.xlabel("Eigenvector value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)

                pdf.savefig()
                plt.close()

                f5 = plt.figure(figsize=(8.5, 11), dpi=400)
                if (g_count2 > 2):
                    sy2 = 2
                    fnt = font2
                else:
                    sy2 = 1
                    fnt = font1
                index = 1
                if Do_kdist:
                    f5.add_subplot(sy2, 2, index)
                    bins4 = np.arange(0.5, max(w_klist) + 1.5, 1)
                    try:
                        wk_sig = str(round(stdev(w_klist), 3))
                    except:
                        wk_sig = "N/A"
                    wk_txt = "Weighted Degree: $\sigma$=" + wk_sig
                    plt.hist(w_klist, bins=bins4)
                    plt.title(wk_txt, fontdict=fnt)
                    plt.xlabel("Degree", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if Do_BCdist:
                    f5.add_subplot(sy2, 2, index)
                    bins5 = np.linspace(min(w_BCdist), max(w_BCdist), 50)
                    plt.hist(w_BCdist, bins=bins5)
                    try:
                        wBC_sig = str(round(stdev(w_BCdist), 3))
                    except:
                        wBC_sig = "N/A"
                    wBC_txt = "Width-Weighted Betweeness: $\sigma$=" + wBC_sig
                    plt.title(wBC_txt, fontdict=fnt)
                    plt.xlabel("Betweenness value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if Do_CCdist:
                    f5.add_subplot(sy2, 2, index)
                    bins6 = np.linspace(min(w_CCdist), max(w_CCdist), 50)
                    try:
                        wCC_sig = str(round(stdev(w_CCdist), 3))
                    except:
                        wCC_sig = "N/A"
                    wCC_txt = "Length-Weighted Closeness: $\sigma$=" + wCC_sig
                    plt.hist(w_CCdist, bins=bins6)
                    plt.title(wCC_txt, fontdict=fnt)
                    plt.xlabel("Closeness value", fontdict=fnt)
                    plt.xticks(fontsize=8)
                    plt.ylabel("Counts", fontdict=fnt)
                    index += 1
                if Do_ECdist:
                    f5.add_subplot(sy2, 2, index)
                    bins7 = np.linspace(min(w_ECdist), max(w_ECdist), 50)
                    plt.hist(w_ECdist, bins=bins7)
                    try:
                        wEC_sig = str(round(stdev(w_ECdist), 3))
                    except:
                        wEC_sig = "N/A"
                    wEC_txt = "Width-Weighted Eigenvector Cent.: $\sigma$=" + wEC_sig
                    plt.title(wEC_txt, fontdict=fnt)
                    plt.xlabel("Eigenvetor value", fontdict=fnt)
                    plt.ylabel("Counts", fontdict=fnt)

                pdf.savefig()
                plt.close()

            if heatmap:
                sz = 30
                lw = 1.5
                update_label("Generating heat maps...")
                time.sleep(0.5)
                if (Do_kdist == 1):
                    f6a = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6a.add_subplot(1, 1, 1)
                    plt.imshow(src, cmap='gray')
                    nodes = G.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=klist, cmap='plasma')
                    if multigraph:
                        for (s, e) in G.edges():
                            for k in range(int(len(G[s][e]))):
                                ge = G[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in G.edges():
                            ge = G[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Degree Heatmap', fontdict=font1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (Do_kdist == 1 and weighted == 1):
                    f6b = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6b.add_subplot(1, 1, 1)
                    plt.imshow(src, cmap='gray')
                    nodes = G.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_klist, cmap='plasma')
                    if multigraph:
                        for (s, e) in G.edges():
                            for k in range(int(len(G[s][e]))):
                                ge = G[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in G.edges():
                            ge = G[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Weighted Degree Heatmap', fontdict=font1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (Do_clust == 1 and multigraph == 0):
                    f6c = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6c.add_subplot(1, 1, 1)
                    plt.imshow(src, cmap='gray')
                    nodes = G.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=Tlist, cmap='plasma')
                    if multigraph:
                        for (s, e) in G.edges():
                            for k in range(int(len(G[s][e]))):
                                ge = G[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in G.edges():
                            ge = G[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Clustering Coefficient Heatmap', fontdict=font1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (Do_BCdist == 1 and multigraph == 0):
                    f6d = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6d.add_subplot(1, 1, 1)
                    plt.imshow(src, cmap='gray')
                    nodes = G.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=BCdist, cmap='plasma')
                    if multigraph:
                        for (s, e) in G.edges():
                            for k in range(int(len(G[s][e]))):
                                ge = G[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in G.edges():
                            ge = G[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Betweenness Centrality Heatmap', fontdict=font1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (Do_BCdist == 1 and weighted == 1 and multigraph == 0):
                    f6e = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6e.add_subplot(1, 1, 1)
                    plt.imshow(src, cmap='gray')
                    nodes = G.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_BCdist, cmap='plasma')
                    if multigraph:
                        for (s, e) in G.edges():
                            for k in range(int(len(G[s][e]))):
                                ge = G[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in G.edges():
                            ge = G[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Width-Weighted Betweenness Centrality Heatmap', fontdict=font1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (Do_CCdist == 1):
                    f6f = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6f.add_subplot(1, 1, 1)
                    plt.imshow(src, cmap='gray')
                    nodes = G.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=CCdist, cmap='plasma')
                    if multigraph:
                        for (s, e) in G.edges():
                            for k in range(int(len(G[s][e]))):
                                ge = G[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in G.edges():
                            ge = G[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Closeness Centrality Heatmap', fontdict=font1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (Do_CCdist == 1 and weighted == 1):
                    f6f = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6f.add_subplot(1, 1, 1)
                    plt.imshow(src, cmap='gray')
                    nodes = G.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_CCdist, cmap='plasma')
                    if multigraph:
                        for (s, e) in G.edges():
                            for k in range(int(len(G[s][e]))):
                                ge = G[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in G.edges():
                            ge = G[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Length-Weighted Closeness Centrality Heatmap', fontdict=font1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (Do_ECdist == 1 and multigraph == 0):
                    f6h = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6h.add_subplot(1, 1, 1)
                    plt.imshow(src, cmap='gray')
                    nodes = G.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=ECdist, cmap='plasma')
                    if multigraph:
                        for (s, e) in G.edges():
                            for k in range(int(len(G[s][e]))):
                                ge = G[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in G.edges():
                            ge = G[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Eigenvector Centrality Heatmap', fontdict=font1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()
                if (Do_ECdist == 1 and weighted == 1 and multigraph == 0):
                    f6h = plt.figure(figsize=(8.5, 8.5), dpi=400)
                    f6h.add_subplot(1, 1, 1)
                    plt.imshow(src, cmap='gray')
                    nodes = G.nodes()
                    gn = np.array([nodes[i]['o'] for i in nodes])
                    plt.scatter(gn[:, 1], gn[:, 0], s=sz, c=w_ECdist, cmap='plasma')
                    if multigraph:
                        for (s, e) in G.edges():
                            for k in range(int(len(G[s][e]))):
                                ge = G[s][e][k]['pts']
                                plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    else:
                        for (s, e) in G.edges():
                            ge = G[s][e]['pts']
                            plt.plot(ge[:, 1], ge[:, 0], 'black', linewidth=lw)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Width-Weighted Eigenvector Centrality Heatmap', fontdict=font1)
                    cbar = plt.colorbar()
                    cbar.set_label('Value')
                    pdf.savefig()
                    plt.close()

            f8 = plt.figure(figsize=(8.5, 8.5), dpi=300)
            f8.add_subplot(1, 1, 1)
            plt.text(0.5, 0.5, run_info, horizontalalignment='center', verticalalignment='center')
            plt.xticks([])
            plt.yticks([])
            pdf.savefig()
            plt.close()

        if (Exp_EL == 1):
            if (weighted == 1):
                fields = ['Source', 'Target', 'Weight', 'Length']
                el = nx.generate_edgelist(G, delimiter=',', data=["weight", "length"])
                with open(file2, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(fields)
                    for line in el:
                        line = str(line)
                        row = line.split(',')
                        try:
                            writer.writerow(row)
                        except:
                            None
                csvfile.close()
            else:
                fields = ['Source', 'Target']
                el = nx.generate_edgelist(G, delimiter=',', data=False)
                with open(file2, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(fields)
                    for line in el:
                        line = str(line)
                        row = line.split(',')
                        try:
                            writer.writerow(row)
                        except:
                            None
                csvfile.close()

        # exporting as gephi file
        if (Do_gexf == 1):
            if multigraph:
                # deleting extraneous info and then exporting the final skeleton
                for (x) in G.nodes():
                    del G.nodes[x]['pts']
                    del G.nodes[x]['o']
                for (s, e) in G.edges():
                    for k in range(int(len(G[s][e]))):
                        try:
                            del G[s][e][k]['pts']
                        except KeyError:
                            None

                nx.write_gexf(G, file1)
            else:
                # deleting extraneous info and then exporting the final skeleton
                for (x) in G.nodes():
                    del G.nodes[x]['pts']
                    del G.nodes[x]['o']
                for (s, e) in G.edges():
                    del G[s][e]['pts']
                nx.write_gexf(G, file1)
        """
        pass

    def approx_conductance_by_spectral(self):
        """
            https://doi.org/10.1016/j.procs.2013.09.311

            Conductance is closely approximable via eigenvalue computation,\
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
            fiedler_vector = nx.fiedler_vector(eig_graph)
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
        lpl_mat = deg_mat - adj_mat

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
