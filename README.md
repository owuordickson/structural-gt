# Structural GTb

A software tool that implements structural graph theory and chaos engineering for nano-structures.

## Installation

This specified version is not currently available for download. Therefore, please follow the manual installation instructions provided below:

* Install Python version 3.11 on your computer.
* Get the ```source code``` folder named **'structural-gt'** and save it to your preferred location on your PC.
* Open a terminal application such as CMD. 
* Navigate to the location where you saved the **'structural-gt'** folder using the terminal. 
* Execute the following commands:

```bash
cd structural-gt
python3 -m venv venv_sgt
source venv_sgt/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install .
deactivate
```

## Executing 

To execute the program, please follow these steps:

* Open a terminal application such as CMD.
* Navigate to the **'structural-gt'** folder using the terminal.
* Once inside the folder, execute the following command:

```bash
structural-gt/venv_sgt/bin/StructuralGTb
```