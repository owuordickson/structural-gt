# Structural GTb

A software tool that allows graph theory analysis of nano-structures. This is a modified version of **StructuralGT** initially proposed by Drew A. Vecchio, DOI: [10.1021/acsnano.1c04711](https://pubs.acs.org/doi/10.1021/acsnano.1c04711?ref=pdf).

## Installation

## 1. Install as software

* Download link: https://forms.gle/UtFfkGGqRoUjzeL47
* Install and enjoy. We would love to hear from you, please give us feedback.

## 2. Install via source code

Therefore, please follow the manual installation instructions provided below:

* Install Python version 3.13 on your computer.
* Git Clone the branch **DicksoOwuor-GUI** from this repo: ```https://github.com/compass-stc/StructuralGT.git```
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

### 2a. Executing GUI app

To execute the program, please follow these steps:

* Open a terminal application such as CMD.
* Execute the following command:

```bash
StructuralGT
```

### 2b. Executing Terminal app

Before executing ```StructuralGT-cli```, you need to specify these parameters:

* **image file path** or **image directory/folder**: *[required and mutually exclusive]* you can set the file path using ```-f path-to-image``` or set the directory path using ```-d path-to-folder```. If directory path is set, StructuralGT will compute the GT metrics of all the images simultaneously,
* **configuration file path**: *[required]* you can set the path to config file using ```-c path-to-config```,
* **type of GT task**: *[required]* you can either 'extract graph' using ```-t 1``` or compute GT metrics using ```-t 2```,
* **allow auto-scaling** : *[optional]* allows StructuralGT to automatically scale images to an optimal size for computation. You can disable this using ```-s 0```.

To execute the program, please follow these steps:

* Open a terminal application such as CMD.
* Execute the following command:

```bash
StructuralGT-cli -d datasets/ -c datasets/sgt_configs.ini -t 2
```

OR 

```bash
StructuralGT-cli -f datasets/InVitroBioFilm.png -c datasets/sgt_configs.ini -t 2
```

OR

```bash
StructuralGT-cli -f datasets/InVitroBioFilm.png -c datasets/sgt_configs.ini -t 1
```

## References
* Drew A. Vecchio, Samuel H. Mahler, Mark D. Hammig, and Nicholas A. Kotov
ACS Nano 2021 15 (8), 12847-12859. DOI: [10.1021/acsnano.1c04711](https://pubs.acs.org/doi/10.1021/acsnano.1c04711?ref=pdf).