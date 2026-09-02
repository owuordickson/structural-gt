[![Downloads](https://pepy.tech/badge/sgtlib)](https://pepy.tech/project/sgtlib) [![Downloads](https://pepy.tech/badge/sgtlib/week)](https://pepy.tech/project/sgtlib)
![Dependents](https://badgen.net/github/dependents-repo/owuordickson/structural-gt/?icon=github)
[![DOI](https://zenodo.org/badge/739102771.svg)](https://doi.org/10.5281/zenodo.16542144)
![Dependents](https://badgen.net/github/license/owuordickson/structural-gt/?icon=github)

# StructuralGT

A software tool that allows graph theory analysis of nanostructures. This is a modified version of **StructuralGT** initially proposed by Drew A. Vecchio, DOI: [10.1021/acsnano.1c04711](https://pubs.acs.org/doi/10.1021/acsnano.1c04711?ref=pdf).

## Installation

## 1. Install as software

* Download link: https://github.com/owuordickson/structural-gt/releases/tag/v3.6.8
* Install and enjoy. 
* 5 minute YouTube tutorial: https://www.youtube.com/watch?v=bEXaIKnse3g
* We would love to hear from you, please give us feedback.

## 2. Install via pip
* Install Python version 3.14 on your computer.
* Execute the following commands:

```bash
pip install sgtlib
```


## 3. Install via source code

Therefore, please follow the manual installation instructions provided below:

* Install Python version 3.14 on your computer.
* Git Clone this repo: ```https://github.com/owuordickson/structural-gt.git```
* Extract the ```source code``` folder named **'structural-gt'** and save it to your preferred location on your PC.
* Open a terminal application such as CMD. 
* Navigate to the location where you saved the **'structural-gt'** folder using the terminal. 
* Execute the following commands:

```bash
cd structural-gt
pip install --upgrade pip
pip install -r requirements.txt
pip install .
```

## 3. Usage

### 3(a) Executing GUI App

To run the GUI version, please follow these steps:

* Open a terminal application such as CMD.
* Execute the following command:

```bash
StructuralGT
```

### 3(b) Executing Terminal App

Before executing ```StructuralGT-cli```, you need to specify these parameters:

* **image file path** or **image directory/folder**: *[required and mutually exclusive]* you can set the file path using ```-f path-to-image``` or set the directory path using ```-d path-to-folder```. If the directory path is set, StructuralGT will compute the GT metrics of all the images simultaneously,
* **configuration file path**: *[required]* you can set the path to config the file using ```-c path-to-config```. To make it easy, find the file ```sgt_configs.ini``` (in the *''root folder''*) and modify it to capture your GT parameters,
* **type of GT task**: *[required]* you can either 'extract graph' using ```-t 1``` or compute GT metrics using ```-t 2```,
* **output directory**: *[optional]* you can set the folder where the GT results will be stored using ```-o path-to-folder```,
* **allow auto-scaling** : *[optional]* allows StructuralGT to automatically scale images to an optimal size for computation. You can disable this using ```-s 0```.

Please follow these steps to execute:

* Open a terminal application such as CMD.
* Execute the following command:

```bash
StructuralGT-cli -d datasets/ -c datasets/sgt_configs.ini -o results/ -t 2
```

OR 

```bash
StructuralGT-cli -f datasets/InVitroBioFilm.png -c datasets/sgt_configs.ini -t 2
```

OR

```bash
StructuralGT-cli -f datasets/InVitroBioFilm.png -c datasets/sgt_configs.ini -t 1
```

### 3(c) Using Library API
To use ```StructuralGT``` library:
* Make sure you **install via pip**
* Create a **Python** script or **Jupyter Notebook** and import modules as shown:

```python
import matplotlib.pyplot as plt
from sgtlib import modules as sgt

# set paths
img_path = "path/to/image"
cfg_file = "path/to/sgt_configs.ini"  # Optional: leave blank


# Define a function for receiving progress updates
def print_updates(progress_val, progress_msg):
    print(f"{progress_val}: {progress_msg}")


# Create a Network object
ntwk_obj, _ = sgt.ImageProcessor.from_image_file(img_path, config_file=cfg_file)

# Apply image filters according to cfg_file
ntwk_obj.add_listener(print_updates)
ntwk_obj.apply_img_filters()
ntwk_obj.remove_listener(print_updates)

# View images
sel_img_batch = ntwk_obj.selected_batch
bin_images = [obj.img_bin for obj in sel_img_batch.images]
grayscale_images = [obj.img_grayscale for obj in sel_img_batch.images]
plt.imshow(bin_images[0])
plt.axis('off')  # Optional: Turn off axis ticks and labels for a cleaner image display
plt.title('Binary Image')
plt.show()

plt.imshow(grayscale_images[0])
plt.axis('off')  # Optional: Turn off axis ticks and labels for a cleaner image display
plt.title('Grayscale Image')
plt.show()

# Extract graph
ntwk_obj.add_listener(print_updates)
ntwk_obj.build_graph_network()
ntwk_obj.remove_listener(print_updates)

# View graph
net_images = [ntwk_obj.graph_obj.img_ntwk]
plt.imshow(net_images[0])
plt.axis('off')  # Optional: Turn off axis ticks and labels for a cleaner image display
plt.title('Graph Image')
plt.show()

# Compute graph theory metrics
compute_obj = sgt.GraphAnalyzer(ntwk_obj)
sgt.GraphAnalyzer.safe_run_analyzer(compute_obj, print_updates)
print(compute_obj.output_df)

# Save in PDF
sgt.GraphAnalyzer.write_to_pdf(compute_obj)
```


### 3(d) Generating Synthetic Networks

The last button on the ribbon opens [NetworkSynth](https://github.com/WilliamLuminary/NetworkSynth), which builds synthetic networks modelled on an extracted graph. It runs as its own program, so you choose the inputs, the settings and the output folder in its window.

NetworkSynth is a private repository, included here as a git submodule in `networksynth`. Cloning StructuralGT without access to it breaks nothing: the folder stays empty and the synthesis button stays disabled, with a tooltip saying what is missing.

To install it you need a GitHub SSH key with access:

```bash
git submodule update --init --checkout networksynth
git -C networksynth fetch origin dist --depth 1
git -C networksynth checkout -B dist FETCH_HEAD
```

`--checkout` is needed because the submodule is set to `update = none`, so a plain `--init` skips it. The last two lines put you on the newest release, and running them again is how you update later.

You never commit anything for this. `git status` stays clean whatever version you are on.

If NetworkSynth fails, the last lines of its output appear in the SGT Logs window.

Pre-release builds live on a `dist-dev` branch. To try one, fetch that name instead of `dist` in the commands above, and fetch `dist` again to go back. This is for developers, and nothing you do in `networksynth` reaches anyone else unless you commit the pin.

`wandb` is the only package not in `requirements.txt`. NetworkSynth uses it in sweep mode for online tracking. You can also point the button at a different folder or a different Python under `[synthesis-settings]` in `sgt_configs.ini`.


## Contributors ✨

Thanks go to these incredible people:

<a href="https://github.com/owuordickson/structural-gt/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=owuordickson/structural-gt" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
