import configparser
from os import path
import pathlib


def load():
    # Load configuration from file
    config_file = pathlib.Path(__file__).parent.absolute() / "configs.cfg"
    config = configparser.SafeConfigParser()
    config.read(config_file)
    # print(config.sections())

    # Image Path
    datadir = config.get('image', 'datadir')
    imagepath = config.get('image', 'imagepath')
    file_path = path.join(datadir, imagepath)

    return config