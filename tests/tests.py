from StructuralGT import modules as sgt

# set paths
img_path = "path/to/image"
cfg_file = "path/to/sgt_configs.ini"   # Optional: leave blank

# Create a Network object
ntwk_obj, _ = sgt.ImageProcessor.create_imp_object(img_path, config_file=cfg_file)

# Apply image filters according to cfg_file
ntwk_obj.apply_img_filters()

# View images
sel_img_batch = ntwk_obj.get_selected_batch()
bin_images = [obj.img_bin for obj in sel_img_batch.images]
mod_images = [obj.img_mod for obj in sel_img_batch.images]
bin_images[0]

# Extract graph
ntwk_obj.build_graph_network()

# View graph
net_images = [sel_img_batch.graph_obj.img_ntwk]
net_images[0]

# Compute graph theory metrics
compute_obj = sgt.GraphAnalyzer(ntwk_obj)
compute_obj.safe_run_analyzer()
print(compute_obj.output_df)

# Save in PDF
sgt.GraphAnalyzer.write_to_pdf(compute_obj)