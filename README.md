# Structural GTb

A software tool that allows graph theory analysis of nano-structures. This is a modified version of **StructuralGT** initially proposed by Drew A. Vecchio, DOI: [10.1021/acsnano.1c04711](https://pubs.acs.org/doi/10.1021/acsnano.1c04711?ref=pdf).

## Installation

This specified version is not currently available for download via GitHub. Therefore, please follow the manual installation instructions provided below:

### a) Install via source code

* Install Python version 3.13 on your computer.
* Download link https://drive.google.com/file/d/1jhr5w0KhCAbMS9kxbOlean0X2q2HrZy9/view?usp=drive_link
* Download, extract the ```source code``` folder named **'structural-gt'** and save it to your preferred location on your PC.
* Open a terminal application such as CMD. 
* Navigate to the location where you saved the **'structural-gt'** folder using the terminal. 
* Execute the following commands:

```bash
cd structural-gt
pip install --upgrade pip
pip uninstall pyqt6-tools
pip uninstall pyqt6-plugins
pip install -r requirements.txt
pip install .
```

### b) Install as executable

* Install Python version 3.13 on your computer.
* Download link (to be provided)
* Open a terminal application such as CMD. 
* Navigate to the location where you saved the **'.whl'** file using the terminal. 
* Execute the following command:

```bash
pip install structuralgt-2.0.1-win_amd64.whl
```

## Executing program

To execute the program, please follow these steps:

* Open a terminal application such as CMD.
* Execute the following command:

```bash
StructuralGT
```


## References
* Drew A. Vecchio, Samuel H. Mahler, Mark D. Hammig, and Nicholas A. Kotov
ACS Nano 2021 15 (8), 12847-12859. DOI: [10.1021/acsnano.1c04711](https://pubs.acs.org/doi/10.1021/acsnano.1c04711?ref=pdf).