import cv2
import matplotlib.pyplot as plt
from sgtlib import modules as sgt

# set paths
img_path = "../datasets/InVitroBioFilm.png"
cfg_file = ""   # Optional: leave blank

# Define a function for receiving progress updates
def print_updates(progress_obj: sgt.ProgressData):
    print(f"{progress_obj.percent}: {progress_obj.message}")

# Create a Network object
ntwk_obj, _ = sgt.ImageProcessor.from_image_file(img_path, config_file=cfg_file)

# Apply image filters according to cfg_file
ntwk_obj.add_listener(print_updates)
ntwk_obj.apply_img_filters()
ntwk_obj.remove_listener(print_updates)

# View images
orig_image = ntwk_obj.image_2d
bin_image = ntwk_obj.binary_image_2d
mod_image = ntwk_obj.processed_image_3d[0]
plt.imshow(cv2.cvtColor(bin_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Optional: Turn off axis ticks and labels for a cleaner image display
plt.title('Binary Image')
plt.show()

print(f"Original Image Shape: {orig_image.shape}\nBinary Image Shape: {bin_image.shape}\nProcessed Image Shape: {mod_image.shape}")
print(f"Original Image: {orig_image}\nBinary Image: {bin_image}\nProcessed Image:{mod_image}")

plt.imshow(mod_image, cmap='gray')
plt.axis('off')  # Optional: Turn off axis ticks and labels for a cleaner image display
plt.title('Processed Image')
plt.show()

# Extract graph
ntwk_obj.add_listener(print_updates)
ntwk_obj.build_graph_network()
ntwk_obj.remove_listener(print_updates)

# View graph
net_image = ntwk_obj.graph_obj.img_ntwk
plt.imshow(net_image)
plt.axis('off')  # Optional: Turn off axis ticks and labels for a cleaner image display
plt.title('Graph Image')
plt.show()

# Compute graph theory metrics
#compute_obj = sgt.GraphAnalyzer(ntwk_obj)
#sgt.GraphAnalyzer.safe_run_analyzer(compute_obj, print_updates)
#print(compute_obj.results_df)

# Save in PDF
#sgt.GraphAnalyzer.write_to_pdf(compute_obj)