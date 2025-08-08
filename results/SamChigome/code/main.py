"""
1. Make sure you install sgtlib v3.5.0 via: `pip install sgtlib==3.5.0`
2. Run this script to re-generate the results we got.
"""

from src.sgtlib.modules import ExpressGT
sgt = ExpressGT(image_file="../../../datasets/InVitroBioFilm.png", output_dir="../results")
sgt.compute_gt_descriptors()

