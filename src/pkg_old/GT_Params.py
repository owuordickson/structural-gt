"""GT_Params: Calculates and collates graph theory indices from
an input graph. Utilizes the NetworkX and GraphRicciCurvature
libraries of algorithms.

Copyright (C) 2021, The Regents of the University of Michigan.

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contributers: Drew Vecchio, Samuel Mahler, Mark D. Hammig, Nicholas A. Kotov
Contact email: vecdrew@umich.edu
"""

from __main__ import *
import networkx as nx
import pandas as pd
import numpy as np
import scipy as sp
import settings
import math
from time import sleep
from sklearn.cluster import spectral_clustering
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality, eigenvector_centrality
from networkx.algorithms import average_node_connectivity, global_efficiency, clustering, average_clustering
from networkx.algorithms import degree_assortativity_coefficient
from networkx.algorithms.flow import maximum_flow
from networkx.algorithms.distance_measures import diameter, periphery
from networkx.algorithms.wiener import wiener_index


def run_GT_calcs(G, Do_kdist, Do_dia, Do_BCdist, Do_CCdist, Do_ECdist, Do_GD, Do_Eff, Do_clust, \
                 Do_ANC, Do_Ast, Do_WI, multigraph):

    # getting nodes and edges and defining variables for later use
    klist = [0]
    Tlist = [0]
    BCdist = [0]
    CCdist = [0]
    ECdist = [0]
    data_dict = {"x":[], "y":[]}

    if multigraph:
        Do_BCdist = 0
        Do_ECdist = 0
        Do_clust = 0

    nnum = int(nx.number_of_nodes(G))
    enum = int(nx.number_of_edges(G))

    if Do_ANC | Do_dia:
        connected_graph = nx.is_connected(G)

    data_dict["x"].append("Number of nodes")
    data_dict["y"].append(nnum)

    data_dict["x"].append("Number of edges")
    data_dict["y"].append(enum)

    settings.progress(35)

    # calculating parameters as requested

    # creating degree histogram
    if(Do_kdist == 1):
        settings.update_label("Calculating degree...")
        klist1 = nx.degree(G)
        ksum = 0
        klist = np.zeros(len(klist1))
        for j in range(len(klist1)):
            ksum += klist1[j]
            klist[j] = klist1[j]
        k = ksum/len(klist1)
        k = round(k, 5)
        data_dict["x"].append("Average degree")
        data_dict["y"].append(k)

    settings.progress(40)

    # calculating network diameter
    if(Do_dia ==1):
        settings.update_label("Calculating diameter...")
        if connected_graph:
            dia = int(diameter(G))
        else:
            dia = 'NaN'
        data_dict["x"].append("Network diameter")
        data_dict["y"].append(dia)

    settings.progress(45)

    # calculating graph density
    if(Do_GD == 1):
        settings.update_label("Calculating density...")
        GD = nx.density(G)
        GD = round(GD, 5)
        data_dict["x"].append("Graph density")
        data_dict["y"].append(GD)

    settings.progress(50)

    # calculating global efficiency
    if (Do_Eff == 1):
        settings.update_label("Calculating efficiency...")
        Eff = global_efficiency(G)
        Eff = round(Eff, 5)
        data_dict["x"].append("Global efficiency")
        data_dict["y"].append(Eff)

    if (Do_WI == 1):
        settings.update_label("Calculating WI...")
        WI = wiener_index(G)
        WI = round(WI, 1)
        data_dict["x"].append("Wiener Index")
        data_dict["y"].append(WI)

    settings.progress(55)

    # calculating clustering coefficients
    if(Do_clust == 1):
        settings.update_label("Calculating clustering...")
        sleep(5)
        Tlist1 = clustering(G)
        Tlist = np.zeros(len(Tlist1))
        for j in range(len(Tlist1)):
            Tlist[j] = Tlist1[j]
        clust = average_clustering(G)
        clust = round(clust, 5)
        data_dict["x"].append("Average clustering coefficient")
        data_dict["y"].append(clust)

    settings.progress(60)

    # calculating average nodal connectivity

    if (Do_ANC == 1):
        settings.update_label("Calculating connectivity...")
        if connected_graph:
            ANC = average_node_connectivity(G)
            ANC = round(ANC, 5)
        else:
            ANC = 'NaN'
        data_dict["x"].append("Average nodal connectivity")
        data_dict["y"].append(ANC)

    settings.progress(65)

    # calculating assortativity coefficient
    if (Do_Ast == 1):
        settings.update_label("Calculating assortativity...")
        Ast = degree_assortativity_coefficient(G)
        Ast = round(Ast, 5)
        data_dict["x"].append("Assortativity coefficient")
        data_dict["y"].append(Ast)

    settings.progress(70)

    # calculating betweenness centrality histogram
    if (Do_BCdist == 1):
        settings.update_label("Calculating betweenness...")
        BCdist1 = betweenness_centrality(G)
        Bsum = 0
        BCdist = np.zeros(len(BCdist1))
        for j in range(len(BCdist1)):
            Bsum += BCdist1[j]
            BCdist[j] = BCdist1[j]
        Bcent = Bsum / len(BCdist1)
        Bcent = round(Bcent, 5)
        data_dict["x"].append("Average betweenness centrality")
        data_dict["y"].append(Bcent)

    settings.progress(75)

    # calculating closeness centrality
    if(Do_CCdist == 1):
        settings.update_label("Calculating closeness...")
        CCdist1 = closeness_centrality(G)
        Csum = 0
        CCdist = np.zeros(len(CCdist1))
        for j in range(len(CCdist1)):
            Csum += CCdist1[j]
            CCdist[j] = CCdist1[j]
        Ccent = Csum / len(CCdist1)
        Ccent = round(Ccent, 5)
        data_dict["x"].append("Average closeness centrality")
        data_dict["y"].append(Ccent)

    settings.progress(80)

    # calculating eigenvector centrality
    if(Do_ECdist == 1):
        settings.update_label("Calculating eigenvector...")
        try:
            ECdist1 = eigenvector_centrality(G, max_iter=100)
        except:
            ECdist1 = eigenvector_centrality(G, max_iter=10000)
        Esum = 0
        ECdist = np.zeros(len(ECdist1))
        for j in range(len(ECdist1)):
            Esum += ECdist1[j]
            ECdist[j] = ECdist1[j]
        Ecent = Esum / len(ECdist1)
        Ecent = round(Ecent, 5)
        data_dict["x"].append("Average eigenvector centrality")
        data_dict["y"].append(Ecent)

    # calculating graph conductance
    settings.update_label("Calculating graph conductance...")
    # conductance_value = compute_all_subsets_conductance(G, 100)
    # conductance_value = approx_conductance_eigenvalues(G)
    # conductance_value = approx_conductance_by_spectral(G)
    # data_dict["x"].append("Graph Conductance")
    # data_dict["y"].append(conductance_value)
    res = approx_conductance_by_spectral(G)
    for item in res:
        data_dict["x"].append(item["name"])
        data_dict["y"].append(item["value"])

    data = pd.DataFrame(data_dict)

    return data, klist, Tlist, BCdist, CCdist, ECdist


def run_weighted_GT_calcs(G, Do_kdist, Do_BCdist, Do_CCdist, Do_ECdist, Do_ANC, Do_Ast, Do_WI, multigraph):

    settings.update_label("Performing weighted analysis...")

    # includes weight in the calculations
    klist = [0]
    BCdist = [0]
    CCdist = [0]
    ECdist = [0]
    if multigraph:
        Do_BCdist = 0
        Do_ECdist = 0
        Do_ANC = 0

    wdata_dict = {"x": [], "y": []}

    if Do_ANC:
        connected_graph = nx.is_connected(G)

    if(Do_kdist == 1):
        klist1 = nx.degree(G, weight='weight')
        ksum = 0
        klist = np.zeros(len(klist1))
        for j in range(len(klist1)):
            ksum += klist1[j]
            klist[j] = klist1[j]
        k = ksum/len(klist1)
        k = round(k, 5)
        wdata_dict["x"].append("Weighted average degree")
        wdata_dict["y"].append(k)


    if (Do_WI == 1):
        WI = wiener_index(G, weight='length')
        WI = round(WI, 1)
        wdata_dict["x"].append("Length-weighted Wiener Index")
        wdata_dict["y"].append(WI)

    if (Do_ANC == 1):
        if connected_graph:
            max_flow = float(0)
            p = periphery(G)
            q = len(p) - 1
            for s in range(0, q - 1):
                for t in range(s + 1, q):
                    flow_value = maximum_flow(G, p[s], p[t], capacity='weight')[0]
                    if (flow_value > max_flow):
                        max_flow = flow_value
            max_flow = round(max_flow, 5)
        else:
            max_flow = 'NaN'
        wdata_dict["x"].append("Max flow between periphery")
        wdata_dict["y"].append(max_flow)

    if (Do_Ast == 1):
        Ast = degree_assortativity_coefficient(G, weight = 'pixel width')
        Ast = round(Ast, 5)
        wdata_dict["x"].append("Weighted assortativity coefficient")
        wdata_dict["y"].append(Ast)

    if(Do_BCdist == 1):
        BCdist1 = betweenness_centrality(G, weight='weight')
        Bsum = 0
        BCdist = np.zeros(len(BCdist1))
        for j in range(len(BCdist1)):
            Bsum += BCdist1[j]
            BCdist[j] = BCdist1[j]
        Bcent = Bsum / len(BCdist1)
        Bcent = round(Bcent, 5)
        wdata_dict["x"].append("Width-weighted average betweenness centrality")
        wdata_dict["y"].append(Bcent)

    if(Do_CCdist == 1):
        CCdist1 = closeness_centrality(G, distance='length')
        Csum = 0
        CCdist = np.zeros(len(CCdist1))
        for j in range(len(CCdist1)):
            Csum += CCdist1[j]
            CCdist[j] = CCdist1[j]
        Ccent = Csum / len(CCdist1)
        Ccent = round(Ccent, 5)
        wdata_dict["x"].append("Length-weighted average closeness centrality")
        wdata_dict["y"].append(Ccent)

    if (Do_ECdist == 1):
        try:
            ECdist1 = eigenvector_centrality(G, max_iter=100, weight='weight')
        except:
            ECdist1 = eigenvector_centrality(G, max_iter=10000, weight='weight')
        Esum = 0
        ECdist = np.zeros(len(ECdist1))
        for j in range(len(ECdist1)):
            Esum += ECdist1[j]
            ECdist[j] = ECdist1[j]
        Ecent = Esum / len(ECdist1)
        Ecent = round(Ecent, 5)
        wdata_dict["x"].append("Width-weighted average eigenvector centrality")
        wdata_dict["y"].append(Ecent)

    wdata = pd.DataFrame(wdata_dict)

    return wdata, klist, BCdist, CCdist, ECdist


def approx_conductance_by_spectral(graph):
    """
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

    data = []

    # 1. Make a copy of the graph
    # eig_graph = graph.copy()

    # 2a. Remove self-looping edges
    eig_graph = remove_self_loops(graph)

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
        non_directed_graph = make_graph_symmetrical(eig_graph)
        try:
            nx.fiedler_vector(non_directed_graph)
        except nx.NetworkXNotImplemented:
            print("Graph is directed. Cannot compute conductance")
            return None
        except nx.NetworkXError:
            # Graph has less than two nodes or is not connected.
            sub_graph_largest, sub_graph_smallest, size = graph_components(eig_graph)
            eig_graph = sub_graph_largest
            data.append({"name": "Subgraph Count", "value": size})
            data.append({"name": "Large Subgraph Node Count", "value": sub_graph_largest.number_of_nodes()})
            data.append({"name": "Large Subgraph Edge Count", "value": sub_graph_largest.number_of_edges()})
            data.append({"name": "Small Subgraph Node Count", "value": sub_graph_smallest.number_of_nodes()})
            data.append({"name": "Small Subgraph Edge Count", "value": sub_graph_smallest.number_of_edges()})
    except nx.NetworkXError:
        # Graph has less than two nodes or is not connected.
        sub_graph_largest, sub_graph_smallest, size = graph_components(eig_graph)
        eig_graph = sub_graph_largest
        data.append({"name": "Subgraph Count", "value": size})
        data.append({"name": "Large Subgraph Node Count", "value": sub_graph_largest.number_of_nodes()})
        data.append({"name": "Large Subgraph Edge Count", "value": sub_graph_largest.number_of_edges()})
        data.append({"name": "Small Subgraph Node Count", "value": sub_graph_smallest.number_of_nodes()})
        data.append({"name": "Small Subgraph Edge Count", "value": sub_graph_smallest.number_of_edges()})

    # 4. Compute normalized-laplacian matrix
    norm_laplacian_matrix = compute_norm_laplacian_matrix(eig_graph)
    # norm_laplacian_matrix = nx.normalized_laplacian_matrix(eig_graph).toarray()

    # 5. Compute eigenvalues
    # e_vals, _ = np.linalg.eig(norm_laplacian_matrix)
    e_vals = sp.linalg.eigvals(norm_laplacian_matrix)

    # 6. Approximate conductance using the 2nd smallest eigenvalue
    eigenvalues = e_vals.real
    val_max, val_min = compute_conductance_range(eigenvalues)
    data.append({"name": "Graph Conductance (max)", "value": val_max})
    data.append({"name": "Graph Conductance (min)", "value": val_min})

    # print(val_max)
    # print(val_min)
    return data


def compute_norm_laplacian_matrix(graph):
    """
    Compute normalized-laplacian-matrix

    :param adj_mat:
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

    return sub_graph_largest, sub_graph_smallest, component_count


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


def approx_conductance_eigenvalues(graph):
    """
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

    temp_adj_mat = nx.adjacency_matrix(graph).todense()

    # 1. Symmetric-ize the Adjacency matrix
    # temp_adj_mat = np.maximum(temp_adj_mat, temp_adj_mat.transpose())

    # 2. Remove (self-loops) non-zero diagonal values from Adjacency matrix
    np.fill_diagonal(temp_adj_mat, 0)

    # 2a. Identify isolated nodes
    graph_symmetric = nx.from_numpy_array(temp_adj_mat)  # create new graph

    # graph_symmetric = graph.copy()
    isolated_nodes = list(nx.isolates(graph_symmetric))

    # 2b. Remove isolated nodes
    graph_symmetric.remove_nodes_from(isolated_nodes)

    # 3a. Identify connected components
    connected_components = list(nx.connected_components(graph))

    # 3b. Find the largest connected component
    largest_component = max(connected_components, key=len)

    # 3c. Create a new graph containing only the largest connected component
    sub_graph_largest = graph.subgraph(largest_component)

    fiedler_vector = nx.fiedler_vector(sub_graph_largest)  # does not work is sub-graphs are disconnected

    laplacian_matrix = nx.normalized_laplacian_matrix(sub_graph_largest).toarray()
    # adjacency_matrix = nx.adjacency_matrix(sub_graph_largest).toarray()
    eigenvalues = np.linalg.eigvals(laplacian_matrix)
    # eigenvalues = nx.normalized_laplacian_spectrum(sub_graph_largest)  # NOT ACCURATE

    # Remove duplicates and sort the eigenvalues in ascending order
    # vals = set(eigenvalues)
    # sorted_vals = np.array(list(vals))
    sorted_vals = np.array(eigenvalues)
    sorted_vals.sort()

    # Sort the eigenvalues in descending order
    # eigenvalues[::-1].sort()

    # approximate conductance using the 2nd smallest eigenvalue
    # conductance_val_max = math.sqrt(sorted_vals[1])
    conductance_val_min = sorted_vals[1] / 2

    # print(temp_adj_mat)
    # print(laplacian_matrix)
    # print(eigenvalues)
    print(sorted_vals)
    print(fiedler_vector)
    # print(conductance_val)
    # print(conductance_val_max)
    print(conductance_val_min)
    return conductance_val_min


def old_approx_conductance_by_spectral(graph):
    """
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
    # meaning that it is: (asymmetric) with self-looping nodes (non-zero diag)
    # adj_mat = np.array([[0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0]])
    temp_adj_mat = nx.adjacency_matrix(graph).todense()

    # 1. Symmetric-ize the Adjacency matrix
    temp_adj_mat = np.maximum(temp_adj_mat, temp_adj_mat.transpose())

    # 2. Remove non-zero diagonal values from Adjacency matrix
    np.fill_diagonal(temp_adj_mat, 0)

    # 2a. Identify isolated nodes
    sp_graph = nx.from_numpy_array(temp_adj_mat)  # create new graph
    # sp_graph = graph.copy()
    isolated_nodes = list(nx.isolates(sp_graph))

    # 2b. Remove isolated nodes
    sp_graph.remove_nodes_from(isolated_nodes)
    adj_mat = nx.adjacency_matrix(sp_graph).todense()

    # 3a. Compute Degree matrix
    deg_mat = np.diag(np.sum(adj_mat, axis=1))
    # print(deg_mat)

    # 3b. Compute Identity matrix
    id_mat = np.identity(adj_mat.shape[0])

    # 3c. Compute (Degree inverse squared) D^{-1/2} matrix
    # Check for singular matrices
    if np.any(np.diag(deg_mat) == 0):
        # Graph has nodes with zero degree. Cannot compute inverse square root of degree matrix.
        # raise ValueError("Graph has nodes with zero degree. Cannot compute inverse square root of degree matrix.")
        print("Graph has nodes with zero degree. Cannot compute conductance")
        return None
    deg_inv_sqrt = np.linalg.inv(np.sqrt(deg_mat))

    # 3d. Compute Laplacian matrix
    lpl_mat = deg_mat - adj_mat

    # 3e. Compute normalized-Laplacian matrix
    norm_lpl_mat = id_mat - np.dot(deg_inv_sqrt, np.dot(adj_mat, deg_inv_sqrt))
    # norm_lpl_mat = np.eye(sp_graph.number_of_nodes()) - np.dot(np.dot(deg_inv_sqrt, adj_mat), deg_inv_sqrt)

    # 8. Compute eigenvalues
    e_vals, _ = np.linalg.eig(norm_lpl_mat)
    e_vals = sp.linalg.eigvals(norm_lpl_mat)

    # 9. Remove duplicates and sort the eigenvalues in ascending order
    eigenvalues = e_vals
    # round_vals = np.round(eigenvalues, decimals=5)
    # vals = set(eigenvalues)
    sorted_vals = np.array(eigenvalues)
    sorted_vals.sort()

    # 10. Approximate conductance using the 2nd smallest eigenvalue
    # conductance_val_max = math.sqrt(sorted_vals[1])
    conductance_val_min = sorted_vals[1] / 2
    # print(eigenvalues)
    print(sorted_vals)
    # print(conductance_val)
    # print(conductance_val_max)
    print(conductance_val_min)
    return conductance_val_min


def calculate_conductance_by_clustering(graph, num_clusters):
    """
    Calculate conductance based on a clustering of the graph.

    Parameters:
    - graph: NetworkX Graph
    - num_clusters: int value of number of clusters

    Returns:
    - conductance: Float, conductance value
    """
    total_cut_edges = 0
    # total_internal_edges = 0
    num_total_edges = graph.number_of_edges()

    # Use spectral clustering to get cluster assignments
    num_clusters = 5  # You can adjust the number of clusters as needed
    cluster_labels = spectral_clustering(nx.to_numpy_array(graph), n_clusters=num_clusters)

    for edge in graph.edges():
        if cluster_labels[edge[0]] != cluster_labels[edge[1]]:
            total_cut_edges += 1
        # else:
        #    total_internal_edges += 1

    conductance = total_cut_edges / (2 * num_total_edges)
    return conductance


def compute_all_subsets_conductance(graph, max_iter):
    """
    Compute conductance for all non-majority subsets of the vertex set.

    Parameters:
    - graph: NetworkX Graph

    Returns:
    - conductance_dict: Dictionary containing conductance values for each subset
    """
    conductance_dict = {}

    # Get the vertex set
    vertex_set = set(graph.nodes())
    min_conductance = np.inf
    i = 0

    # Iterate through all possible subsets
    connected_components = list(nx.connected_components(graph))
    for component in connected_components:
        try:
            # Calculate conductance for the subset
            conductance_value = nx.conductance(graph, component)
            if conductance_value < min_conductance:
                min_conductance = conductance_value
            conductance_dict[component] = conductance_value
            print(conductance_value)
        except ZeroDivisionError:
            pass
        if i >= max_iter:
            return min_conductance
        i += 1
        # complement_set = set(vertex_set) - set(subset)
        # Uncomment the following line if you also want conductance for the complement set
        # conductance_dict[complement_set] = nx.conductance(graph, list(complement_set))

    # for r in range(1, len(vertex_set) // 2 + 1):  # Consider subsets of size up to half the vertex set
    #    for subset in itertools.combinations(vertex_set, r):

    print(conductance_dict)
    return min_conductance
