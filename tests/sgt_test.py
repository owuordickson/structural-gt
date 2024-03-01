import sgt
import numpy as np
import networkx as nx

try:
    mat = np.array([[0, 1, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 1], [1, 0, 1, 1, 0]])
    nx_g = nx.from_numpy_array(mat)
    adj_mat = nx.to_numpy_array(nx_g)
    size = np.size(adj_mat)
    flat_mat = np.ravel(adj_mat, order='C')
    str_mat = np.array2string(flat_mat, precision=2, separator=' ', suppress_small=True)
    str_mat = str_mat.replace('[', '').replace(']', '')

    # str_mat = "0 1 0 0 1 1 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 1 1 0"
    anc = sgt.compute_anc(str_mat, size, 8, 1)
    print("Average Node Connectivity: " + str(anc))
    # print(nx_g)
    # print(adj_mat)
    # print(size)
    # print(flat_mat)
    # print(str_mat)
except Exception as err:
    print(err)
