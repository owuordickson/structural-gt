def compute_betweenness_centrality(self):
    """
        Implements ideas proposed in: https://doi.org/10.1016/j.socnet.2004.11.009

        Computes betweenness centrality by also considering the edges that all paths (and not just the shortest path)\
        that passes through a vertex. The proposed idea is referred to as: 'random walk betweenness'.

        Random walk betweenness centrality is computed from a fully connected parts of the graph, because each \
        iteration of a random walk must move from source node to destination without disruption. Therefore, if a graph\
        is composed of isolated sub-graphs then betweenness centrality will be limited to only the fully connected\
        sections of the graph. An average is computed after an iteration of x random walks along edges.

        This measure is already implemented in 'networkx' package. Here is the link:\
        https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.current_flow_betweenness_centrality_subset.html

        :return:
        """
    """
        # (NOT TRUE) Important Note: the graph CANNOT have isolated nodes or isolated sub-graphs
        # (NOT TRUE) Note: only works with fully-connected graphs
        # So, random betweenness centrality is computed between source node and destination node.

        graph = self.gc.nx_graph

        # 1. Compute laplacian matrix L = D - A
        lpl_mat = nx.laplacian_matrix(graph).toarray()
        print(lpl_mat)

        # 2. Remove any single row and corresponding column from L
        # 3. Invert the resulting matrix
        # 4. Add back the removed row and column to form matrix T
        # 5. Calculate betweenness from T
        """


x = (10**-9)
y = 1e-9
# print(x == y)

filename = 'file.qptiff'
ALLOWED_IMG_EXTENSIONS = ['*.jpg', '*.png', '*.jpeg', '*.tif', '*.qptiff']
pattern_string = ' '.join(ALLOWED_IMG_EXTENSIONS)
img_ext = ALLOWED_IMG_EXTENSIONS.copy()
# print(pattern_string)
print(img_ext)


"""
import os
from PIL import Image
img_dir_path = "../datasets/at3_10/"
files = os.listdir(img_dir_path)
files = sorted(files)
# Open all images
images = [Image.open(os.path.join(str(img_dir_path), img)) for img in files]
# Save as a multi-page TIFF
images[0].save("combined_stack.tiff", save_all=True, append_images=images[1:])
"""
