import os
import sgt
import numpy as np
import networkx as nx

try:
    mat = np.array([[0, 1, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 1], [1, 0, 1, 1, 0]])
    nx_g = nx.from_numpy_array(mat)
    # adj_mat = nx.to_numpy_array(nx_g)
    adj_mat = nx.adjacency_matrix(nx_g).todense()
    size = np.size(adj_mat)
    flat_mat = np.ravel(adj_mat, order='C')
    str_mat = str(flat_mat.tolist()).replace('[', '').replace(']', '')

    # str_mat = "0 1 0 0 1 1 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 1 1 0"
    # anc = sgt.compute_anc(str_mat, size, 8, 1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = 'graph.txt'
    graph_file = os.path.join(script_dir, graph_path)
    nx.write_edgelist(nx_g, graph_file, data=False)
    anc = sgt.compute_anc(graph_file, size, 8, 1)

    print("Average Node Connectivity: " + str(anc))
    # print(nx_g)
    # print(adj_mat)
    # print(size)
    # print(flat_mat)
    # print(str_mat)
except Exception as err:
    print(err)
