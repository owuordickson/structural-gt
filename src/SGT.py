# SPDX-License-Identifier: GNU GPL v3

"""
A launcher for executing the application as a Window app or a Terminal app.
"""


from StructuralGT.configs.config_loader import detect_cuda_and_install_cupy
from StructuralGT.entrypoints import main_gui

if __name__ == "__main__":
    detect_cuda_and_install_cupy()
    main_gui()
