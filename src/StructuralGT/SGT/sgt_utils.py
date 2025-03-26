import os
import io
import csv
import base64
import multiprocessing as mp
from typing import LiteralString
from PIL import Image



def get_num_cores():
    """
    Finds the count of CPU cores in a computer or a SLURM super-computer.
    :return: number of cpu cores (int)
    """
    num_cores = __get_slurm_cores__()
    if not num_cores:
        num_cores = mp.cpu_count()
    return num_cores


def __get_slurm_cores__():
    """
    Test computer to see if it is a SLURM environment, then gets number of CPU cores.
    :return: count of CPUs (int) or False
    """
    try:
        cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        return cores
    except ValueError:
        try:
            str_cores = str(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            temp = str_cores.split('(', 1)
            cpus = int(temp[0])
            str_nodes = temp[1]
            temp = str_nodes.split('x', 1)
            str_temp = str(temp[1]).split(')', 1)
            nodes = int(str_temp[0])
            cores = cpus * nodes
            return cores
        except ValueError:
            return False
    except KeyError:
        return False


def write_txt_file(data: str, path: LiteralString | str | bytes, wr=True):
    """Description
        Writes data into a txt file.

        :param data: information to be written
        :param path: name of file and storage path
        :param wr: writes data into file if True
        :return:
    """
    if wr:
        with open(path, 'w') as f:
            f.write(data)
            f.close()
    else:
        pass


def write_csv_file(csv_file: LiteralString | str | bytes, column_tiles: list, data):
    """
    Write data to a csv file.
    Args:
        csv_file: name of the csv file
        column_tiles: list of column names
        data: list of data
    Returns:

    """
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(column_tiles)
        for line in data:
            line = str(line)
            row = line.split(',')
            try:
                writer.writerow(row)
            except csv.Error:
                pass
    csvfile.close()


def get_cv_base64(img_cv):
    """ Converts an OpenCV image to a base64 string."""
    img_pil = Image.fromarray(img_cv)  # Convert to PIL Image
    return pil_to_base64(img_pil)


def pil_to_base64(img_pil):
    """Convert a PIL Image to a base64 string."""
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")  # Save image to buffer
    buffer.seek(0)
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")  # Convert to Base64 string
    return base64_data
