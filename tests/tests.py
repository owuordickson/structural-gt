import matplotlib.pyplot as plt
# from src.StructuralGT.Networks.binarizer import Binarizer
from src.StructuralGT.Networks.fiber_network import FiberNetwork
from src.StructuralGT.Networks.structural import Degree


# 1. IMAGE PROCESSING
# Nanowires = Network('dataset/Nanowires')
# B = Binarizer('dataset/Nanowires/Nanowires.tif')
# Nanowires.binarize()


# AdaptiveThreshold = {"Thresh_method": 1, "gamma": 1.001, "md_filter": 0, "g_blur": 0, "autolvl": 0, "fg_color": 0, "laplacian": 0, "scharr": 0, "sobel": 0, "lowpass": 0, "asize": 55, "bsize": 1, "wsize": 1, "thresh": 128.0}
# Nanowires_AdaptiveThreshold = FiberNetwork('../datasets/Nanowires', binarized_dir='AdaptiveThreshold')
# Nanowires_AdaptiveThreshold.binarize(options=AdaptiveThreshold)


# 2. GRAPH EXTRACTION
GaussianBlur = {"Thresh_method": 0, "gamma": 1.001, "md_filter": 0, "g_blur": 1, "autolvl": 0, "fg_color": 0, "laplacian": 0, "scharr": 0, "sobel": 0,"lowpass": 0, "asize": 3, "bsize": 11, "wsize": 1, "thresh": 128.0}
Nanowires = FiberNetwork('../datasets/Nanowires', binarized_dir='GaussianBlur')
# Nanowires.img_to_skel(img_options=GaussianBlur, merge_nodes=True, prune=500, remove_objects=500)
Nanowires.img_to_skel(img_options=GaussianBlur)
plt.imshow(Nanowires.skeleton, cmap='gray')

Nanowires.set_graph()
A = Nanowires.graph.get_adjacency()
Nanowires.graph_plot()


# 3. GRAPH ANALYSIS
D = Degree()
D.compute(Nanowires)
print(f'Average degree is {D.average_degree}')

Nanowires.node_plot(parameter=D.degree)
