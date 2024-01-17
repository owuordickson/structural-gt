import networkx as nx
import numpy as np
import GT_Params

# Test graph
adj_mat = np.array([[0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0]])
G = nx.from_numpy_array(adj_mat)  # create a graph
GT_Params.approx_conductance_by_spectral(G)

print("\n ...NEXT ... \n")
GT_Params.approx_conductance_eigenvalues(G)
