import matplotlib.pyplot as plt
from sgtlib import modules as sgt

# set paths
img_path = "../datasets/InVitroBioFilm.png"
cfg_file = ""   # Optional: leave blank

# Define a function for receiving progress updates
def print_updates(progress_val, progress_msg):
    print(f"{progress_val}: {progress_msg}")

# Create a Network object
ntwk_obj, _ = sgt.ImageProcessor.create_imp_object(img_path, config_file=cfg_file)

# Apply image filters according to cfg_file
ntwk_obj.add_listener(print_updates)
ntwk_obj.apply_img_filters()
ntwk_obj.remove_listener(print_updates)

# View images
bin_images = ntwk_obj.binary_image_3d
mod_images = ntwk_obj.processed_image_3d
plt.imshow(bin_images[0])
plt.axis('off')  # Optional: Turn off axis ticks and labels for a cleaner image display
plt.title('Binary Image')
plt.show()

plt.imshow(mod_images[0])
plt.axis('off')  # Optional: Turn off axis ticks and labels for a cleaner image display
plt.title('Processed Image')
plt.show()

# Extract graph
ntwk_obj.add_listener(print_updates)
ntwk_obj.build_graph_network()
ntwk_obj.remove_listener(print_updates)

# View graph
net_image = ntwk_obj.graph_image
plt.imshow(net_image)
plt.axis('off')  # Optional: Turn off axis ticks and labels for a cleaner image display
plt.title('Graph Image')
plt.show()

# Compute graph theory metrics
compute_obj = sgt.GraphAnalyzer(ntwk_obj)
sgt.GraphAnalyzer.safe_run_analyzer(compute_obj, print_updates)
print(compute_obj.results_df)

# Save in PDF
sgt.GraphAnalyzer.write_to_pdf(compute_obj)