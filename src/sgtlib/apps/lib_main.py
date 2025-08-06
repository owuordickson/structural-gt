# SPDX-License-Identifier: GNU GPL v3

"""
Implementations for running StructuralGT via PyPi library
"""

from .cli_main import TerminalApp
from ..utils.config_loader import strict_read_config_file
from ..imaging.image_processor import ImageProcessor, ALLOWED_IMG_EXTENSIONS

class ExpressGT:
    """Exposes Terminal app to PyPi."""

    def __init__(self, image_dir: str = None, image_file: str = None, output_dir: str = "", config_file: str = ""):
        """
        Exposes Terminal app methods for executing StructuralGT tasks. Please provide either
        `image_dir` or `image_path`, but not both.

        :param image_dir: Directory containing image files (make sure only the required image files are in this directory).
        :param image_file: Path to the image file.
        :param output_dir: Directory where results files of StructuralGT are stored.
        :param config_file: Path to the configuration file (usually 'sgt_config.ini').
        """
        if (image_dir is None and image_file is None) or (image_dir and image_file):
            raise ValueError("You must provide either `image_path` or `image_dir`, but not both.")
        self._image_dir = image_dir
        self._image_file = image_file
        self._config_file = config_file
        self._output_dir = output_dir

        # Create Terminal App
        self._term_app = TerminalApp(self._config_file)
        self._term_app.start()

    def process_image(self):
        """Runs StructuralGT task that applies the selected filters on the image."""
        self._term_app.task_process_image()

    def extract_graph(self):
        """Run StructuralGT task to extract graph."""
        self._term_app.task_extract_graph()

    def compute_gt_descriptors(self):
        """Run StructuralGT task to compute the selected graph theory descriptors."""
        run_multi_gt = True if self._image_dir != "" else False
        self._term_app.task_compute_multi_gt() if run_multi_gt else self._term_app.task_compute_gt()
