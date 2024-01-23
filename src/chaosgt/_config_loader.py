import configparser
# from os import path
import pathlib


def load():
    # Load configuration from file
    config_file = pathlib.Path(__file__).parent.absolute() / "configs.cfg"
    config = configparser.SafeConfigParser()
    config.read(config_file)
    # print(config.sections())

    # 1. Image Path
    imagepath = config.get('image-dir', 'single_image_path')
    multi_imagepath = config.get('image-dir', 'multi_image_path')
    output_path = config.get('image-dir', 'gt_output_path')

    # 2. Image Detection settings

    # 3. Graph Extraction Settings

    # 4. Networkx Calculation Settings

    return config
