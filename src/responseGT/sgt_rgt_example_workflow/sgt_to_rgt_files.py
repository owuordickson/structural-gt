import os
from StructuralGT.networks import Network
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 1. Setup Project and File Paths ---
project_name = 'example_project'
image_filename = 'example_image.bmp'
image_path = os.path.join(project_name, image_filename)
output_directory = 'Processed_Images'


# --- 2. Define Image Processing Options ---
processing_options = {
    "Thresh_method": 0,
    "thresh": 128.0,
    "gamma": 1.001,
    "md_filter": 0,
    "g_blur": 0,
    "autolvl": 0,
    "fg_color": 1,#1 for black network, 0 for white network
    "laplacian": 0,
    "scharr": 0,
    "sobel": 0,
    "lowpass": 0,
    "asize": 3,
    "bsize": 11,
    "wsize": 1,
}


# --- 3. Create Network Object and Process Image ---
# Make sure the source image actually exists in the project folder
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: The source image was not found at '{image_path}'")

# This object knows to look for images inside the 'system' directory
system_network = Network(project_name, binarized_dir=output_directory)


# The method automatically finds images in the project folder ('system/').
# We only need to provide the processing options.
print(f"Processing images in '{project_name}' directory...")
system_network.binarize(options=processing_options)


# --- 4. Confirmation of Path ---
output_path = os.path.join(project_name, output_directory)
print("\nSuccess! ✔️")
print(f"The processed image has been saved in the '{output_path}' directory.")


# --- 5. Convert Image to Skeleton ---
print("\nGenerating skeleton from binarized image...")
skel_system = Network(project_name, binarized_dir=output_directory)
skel_system.img_to_skel()
#plt.show() #This can be uncommented to show display the skeleton

# --- 6. Generate and Plot the Graph ---
print("\nConverting skeleton to graph...")
# Convert the pixel-based skeleton into a node-and-edge graph
skel_system.set_graph()
skel_system.graph_plot()
print("Displaying graph. Close the image window to end the script.")
plt.show()

# You can get the adjacency matrix if you need it
A = skel_system.graph.get_adjacency()
print(f"Graph generated. Adjacency matrix shape: {A.shape}")

# --- 7. Export Graph Data to CSV Files ---

N=skel_system.Gr.vcount()
verts=skel_system.Gr.vs
vertpos = np.array([verts[n]['o'] for n in range(N)])

eds = skel_system.Gr.es
Ne = len(eds)
edgelist = np.array([[eds[n].source,eds[n].target] for n in range(Ne)])

df = pd.DataFrame(vertpos)

# Export the DataFrame to a CSV file without header and index
df.to_csv('vertexPositions.csv', header=False, index=False)

df = pd.DataFrame(edgelist)

# Export the DataFrame to a CSV file without header and index
df.to_csv('edgeList.csv', header=False, index=False)

















