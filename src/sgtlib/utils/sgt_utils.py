# SPDX-License-Identifier: GNU GPL v3

"""
StructuralGT utility functions.
"""

import os
import io
import sys
import cv2
import json
import base64
# import socket
import logging
# import platform
import dropbox
import requests
import gsd.hoomd
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing as mp
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from cv2.typing import MatLike
from typing import LiteralString
from dataclasses import dataclass
# from cryptography.fernet import Fernet


@dataclass
class ProgressData:
    """
    A data class for sending updates to outside functions.

    Attributes
    ----------
    percent : int
        Progress value, the range is 0–100%.
    message : str
        Progress message to be displayed.
    type : str
        Message type, it can be either: "info", "warning", "error".
    sender : str
        Sender of the message.
    """
    percent: int = -1
    message: str = ""
    type: str = ""  # "info", "warning", "error"
    sender: str = ""


@dataclass
class TaskResult:
    task_id: str = ""
    status: str = ""
    message: str = ""
    data: object|list|None = None


class AbortException(Exception):
    """Custom exception to handle task cancellation initiated by the user or an error."""
    pass


class ProgressUpdate:
    """
    A class for sending updates to outside functions. It uses listener functions to send updates to outside functions.
    """

    def __init__(self):
        """
        A class for sending updates to outside functions.

        Example 1:
        -------
        >>> def print_progress(code, msg):
        ...     print(f"{code}: {msg}")

        >>> upd = ProgressUpdate()
        >>> upd.add_listener(print_progress)  # to get updates
        >>> upd.update_status((1, "Sending update ..."))
        1: Sending update ...
        >>> upd.remove_listener(print_progress)  # to opt out of updates

        Example 2:
        ---------
        >>> def print_progress(p_data: ProgressData):
        ...     print(f"{p_data.percent}: {p_data.message}")

        >>> upd = ProgressUpdate()
        >>> upd.add_listener(print_progress)  # to get updates
        >>> msg_data = ProgressData(percent=1, message="Sending update ...")
        >>> upd.update_status(msg_data)
        1: Sending update ...
        >>> upd.remove_listener(print_progress)  # to opt out of updates
        """
        self.__listeners = []
        self.abort = False

    def abort_tasks(self) -> None:
        """
        Set abort flag.
        :return:
        """
        self.abort = True

    def add_listener(self, func) -> None:
        """
        Add functions from the list of listeners.
        :param func:
        :return:
        """
        if func in self.__listeners:
            return
        self.__listeners.append(func)

    def remove_listener(self, func) -> None:
        """
        Remove functions from the list of listeners.
        :param func:
        :return:
        """
        if func not in self.__listeners:
            return
        self.__listeners.remove(func)

    def update_status(self, args=None) -> None:
        """
        Run all the functions that are saved as listeners.

        :param args:
        :return:
        """
        # Trigger events.
        if args is None:
            args = ()
        if not isinstance(args, (tuple, list)):
            args = (args,)
        for func in self.__listeners:
            func(*args)


def get_num_cores() -> int | bool:
    """
    Finds the count of CPU cores in a computer or a SLURM supercomputer.
    :return: Number of cpu cores (int)
    """

    def __get_slurm_cores__():
        """
        Test the computer to see if it is a SLURM environment, then gets the number of CPU cores.
        :return: Count of CPUs (int) or False
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

    num_cores = __get_slurm_cores__()
    if not num_cores:
        num_cores = mp.cpu_count()
    return int(num_cores)


def verify_path(a_path) -> tuple[bool, str]:
    if not a_path:
        return False, "No folder/file selected."

    # Convert QML "file:///" path format to a proper OS path
    if a_path.startswith("file:///"):
        if sys.platform.startswith("win"):
            # Windows Fix (remove extra '/')
            a_path = a_path[8:]
        else:
            # macOS/Linux (remove "file://")
            a_path = a_path[7:]

    # Normalize the path
    a_path = os.path.normpath(a_path)

    if not os.path.exists(a_path):
        return False, f"File/Folder in {a_path} does not exist. Try again."
    return True, a_path


def install_package(package) -> None:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logging.info(f"Successfully installed {package}", extra={'user': 'SGT Logs'})
    except subprocess.CalledProcessError:
        logging.info(f"Failed to install {package}: ", extra={'user': 'SGT Logs'})


def detect_cuda_version() -> str | None:
    """Check if CUDA is installed and return its version."""
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        if 'release 12' in output:
            return '12'
        elif 'release 11' in output:
            return '11'
        else:
            return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.info(f"Please install 'NVIDIA GPU Computing Toolkit' via: https://developer.nvidia.com/cuda-downloads", extra={'user': 'SGT Logs'})
        return None


"""
def detect_cuda_and_install_cupy():
    try:
        import cupy
        logging.info(f"CuPy is already installed: {cupy.__version__}", extra={'user': 'SGT Logs'})
        return
    except ImportError:
        logging.info("CuPy is not installed.", extra={'user': 'SGT Logs'})

    def is_connected(host="8.8.8.8", port=53, timeout=3):
        # Check if the system has an active internet connection.
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False

    if not is_connected():
        logging.info("No internet connection. Cannot install CuPy.", extra={'user': 'SGT Logs'})
        return

    # Handle macOS (Apple Silicon) - CPU only
    if platform.system() == "Darwin" and platform.processor().startswith("arm"):
        logging.info("Detected MacOS with Apple Silicon (M1/M2/M3). Installing CPU-only version of CuPy.", extra={'user': 'SGT Logs'})
        # install_package('cupy')  # CPU-only version
        return

    # Handle CUDA systems (Linux/Windows with GPU)
    cuda_version = detect_cuda_version()

    if cuda_version:
        logging.info(f"CUDA detected: {cuda_version}", extra={'user': 'SGT Logs'})
        if cuda_version == '12':
            install_package('cupy-cuda12x')
        elif cuda_version == '11':
            install_package('cupy-cuda11x')
        else:
            logging.info("CUDA version not supported. Installing CPU-only CuPy.", extra={'user': 'SGT Logs'})
            install_package('cupy')
    else:
        # No CUDA found, fall back to the CPU-only version
        logging.info("CUDA not found. Installing CPU-only CuPy.", extra={'user': 'SGT Logs'})
        install_package('cupy')

    # Proceed with installation if connected
    cuda_version = detect_cuda_version()
    if cuda_version == '12':
        install_package('cupy-cuda12x')
    elif cuda_version == '11':
        install_package('cupy-cuda11x')
    else:
        logging.info("No CUDA detected or NVIDIA GPU Toolkit not installed. Installing CPU-only CuPy.", extra={'user': 'SGT Logs'})
        install_package('cupy')
"""


def write_txt_file(data: str, path: LiteralString | str | bytes, wr=True) -> None:
    """Description
        Writes data into a txt file.

        :param data: Information to be written
        :param path: name of the file and storage path
        :param wr: writes data into file if True
        :return:
    """
    if wr:
        with open(path, 'w') as f:
            f.write(data)
            f.close()
    else:
        pass


def write_gsd_file(f_name: str, skeleton: np.ndarray) -> None:
    """
    A function that writes graph particles to a GSD file. Visualize with OVITO software.
    Acknowledgements: Alain Kadar (https://github.com/compass-stc/StructuralGT/)

    :param f_name: gsd.hoomd file name
    :param skeleton: skimage.morphology skeleton
    """
    # pos_count = int(sum(skeleton.ravel()))
    particle_positions = np.asarray(np.where(np.asarray(skeleton) != 0)).T
    with gsd.hoomd.open(name=f_name, mode="w") as f:
        s = gsd.hoomd.Frame()
        s.particles.N = len(particle_positions)  # OR pos_count
        s.particles.position = particle_positions
        s.particles.types = ["A"]
        s.particles.typeid = ["0"] * s.particles.N
        f.append(s)


def gsd_to_skeleton(gsd_file: str, is_2d:bool=False) -> None | np.ndarray:
    """
    A function that takes a gsd file and returns a NetworkX graph object.
    Acknowledgements: Alain Kadar (https://github.com/compass-stc/StructuralGT/)

    :param gsd_file: gsd.hoomd file name;
    :param is_2d: is the skeleton 2D?
    :return:
    """

    def shift(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Translates all points such that the minimum coordinate in points is the origin.

        Args:
            points: The points to shift.

        Returns:
            The shifted points.
            The applied shift.
        """
        if is_2d:
            shifted_points = np.full(
                (np.shape(points)[0], 2),
                [np.min(points.T[0]), np.min(points.T[1])],
            )
        else:
            shifted_points = np.full(
                (np.shape(points)[0], 3),
                [
                    np.min(points.T[0]),
                    np.min(points.T[1]),
                    np.min(points.T[2]),
                ],
            )
        points = points - shifted_points
        return points, shifted_points

    def reduce_dim(all_positions: np.ndarray) -> np.ndarray:
        """For lists of positions where all elements along one axis have the same
        value, this returns the same list of positions but with the redundant
        dimension(s) removed.

        Args:
            all_positions: The positions to reduce.

        Returns:
            The reduced positions
        """

        unique_positions = np.asarray(
            list(len(np.unique(all_positions.T[i])) for i in range(len(all_positions.T)))
        )
        redundant = unique_positions == 1
        all_positions = all_positions.T[~redundant].T
        return all_positions

    frame = gsd.hoomd.open(name=gsd_file, mode="r")[0]
    positions = shift(frame.particles.position.astype(int))[0]

    if sum((positions < 0).ravel()) != 0:
        positions = shift(positions)[0]

    if is_2d:
        """
        is_2d (optional, bool):
            Whether the skeleton is 2D. If True it only ensures additional
            redundant axes from the position array is removed. It does not
            guarantee a 3d graph.
        """
        positions = reduce_dim(positions)
        new_pos = np.zeros(positions.T.shape)
        new_pos[0] = positions.T[0]
        new_pos[1] = positions.T[1]
        positions = new_pos.T.astype(int)

    skel_int = np.zeros(
        list((max(positions.T[i]) + 1) for i in list(
            range(min(positions.shape))))
    )
    skel_int[tuple(list(positions.T))] = 1
    return skel_int.astype(int)


def csv_to_graph(csv_path: str) -> None | nx.Graph:
    """
    Load a graph from a file that may contain:
      - Edge list (2 columns)
      - Adjacency matrix (square matrix)
      - XYZ positions (3 columns: x, y, z, edges inferred by distance threshold)

    :param csv_path: Path to the graph file
    """

    # Check if the first line is text (header) instead of numbers
    with open(csv_path, "r") as f:
        first_line = f.readline()
    try:
        [float(x) for x in first_line.replace(",", " ").split()]
        skip = 0  # numeric → no header
    except ValueError:
        skip = 1  # not numeric → skip header

    # Try to read as a numeric matrix
    try:
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, skiprows=skip)
    except ValueError:
        return None

    if data is None:
        return None

    # Case 1: Edge list (two columns)
    if data.ndim == 2 and data.shape[1] == 2:
        nx_graph = nx.Graph()
        for u, v in data.astype(int):
            nx_graph.add_edge(u, v)
        return nx_graph

    # Case 2: Adjacency matrix (square matrix)
    elif data.ndim == 2 and data.shape[0] == data.shape[1]:
        nx_graph = nx.from_numpy_array(data)
        return nx_graph

    # Case 3: XYZ positions (three columns)
    elif data.ndim == 2 and data.shape[1] == 3:
        from scipy.spatial import distance_matrix
        # Build graph based on proximity (set threshold distance)
        threshold = 1.0
        dist_mat = distance_matrix(data, data)
        nx_graph = nx.Graph()
        for i in range(len(data)):
            nx_graph.add_node(i, pos=data[i])
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if dist_mat[i, j] < threshold:
                    nx_graph.add_edge(i, j, weight=dist_mat[i, j])
        return nx_graph
    else:
        return None


def img_to_base64(img: MatLike | Image.Image) -> str:
    """ Converts a Numpy/OpenCV or PIL image to a base64 encoded string."""

    def opencv_to_base64(img_arr: MatLike) -> str:
        """Convert an OpenCV/Numpy image to a base64 string."""
        success, encoded_img = cv2.imencode('.png', img_arr)
        if success:
            buffer = io.BytesIO(encoded_img.tobytes())
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return base64_data
        else:
            return ""

    if img is None:
        return ""

    if type(img) == np.ndarray:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return opencv_to_base64(img_rgb)

    if type(img) == Image.Image:
        # Convert to numpy, apply safe conversion
        np_img = np.array(img)
        img_norm = safe_uint8_image(np_img)
        return opencv_to_base64(img_norm)
    return ""


def plot_to_opencv(fig: plt.Figure) -> MatLike | None:
    """Convert a Matplotlib figure to an OpenCV BGR image (Numpy array), retaining colors."""
    if fig:
        # Save a figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Convert buffer to NumPy array
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        # Decode image including the alpha channel (if any)
        img_cv_rgba = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        # Convert RGBA to RGB if needed
        if img_cv_rgba.shape[2] == 4:
            img_cv_rgb = cv2.cvtColor(img_cv_rgba, cv2.COLOR_RGBA2RGB)
        else:
            img_cv_rgb = img_cv_rgba

        # Convert RGB to BGR to match OpenCV color space
        img_cv_bgr = cv2.cvtColor(img_cv_rgb, cv2.COLOR_RGB2BGR)
        return img_cv_bgr
    return None


def safe_uint8_image(img: MatLike) -> MatLike | None:
    """
    Converts an image to uint8 safely:
        - If already uint8, returns as is.
        - If float or other type, normalizes to 0–255 and converts to uint8.
    """
    if img is None:
        return None

    if img.dtype == np.uint8:
        return img

    # Handle float or other types
    min_val = float(np.min(img))
    max_val = float(np.max(img))

    if min_val == max_val:
        # Avoid divide by zero; return constant grayscale
        return np.full(img.shape, 0 if min_val == 0 else 255, dtype=np.uint8)

    # Normalize to 0–255
    norm_img = ((img - min_val) / (max_val - min_val)) * 255.0
    return norm_img.astype(np.uint8)


def sgt_excel_to_dataframe(excel_dir_path: str, allowed_ext: str = ".xlsx") -> dict[str, pd.DataFrame] | None:
    """
        Loads multiple Excel files generated by the StructuralGT–Scaling Behavior module into Pandas DataFrames.

        This function scans the specified directory for Excel files with the given extension,
        reads each file into a Pandas DataFrame, and stores the results in a dictionary
        where the keys are file names (without extensions).

        Args:
            excel_dir_path (str): Path to the directory containing Excel files
            allowed_ext (str, optional): Allowed file extension (default: ".xlsx")

        Returns:
            dict[str, pd.DataFrame] | None:
                A dictionary mapping each file name (without extension) to its corresponding
                DataFrame, or None if no valid Excel files are found.
    """

    if excel_dir_path is None:
        return None

    files = os.listdir(excel_dir_path)
    files = sorted(files)
    rename_map = {
        "Nodes-Number of edge.": "Nodes-Edges",
        "Nodes-Number of edge. (Fitting)": "Nodes-Edges(Fit)",
        "Nodes-Average degree": "Nodes-Degree",
        "Nodes-Average degree (Fitting)": "Nodes-Degree(Fit)",
        "Nodes-Network diamet.": "Nodes-Diameter",
        "Nodes-Network diamet. (Fitting)": "Nodes-Diameter(Fit)",
        "Nodes-Graph density": "Nodes-Density",
        "Nodes-Graph density (Fitting)": "Nodes-Density(Fit)",
        "Nodes-Average betwee.": "Nodes-BC",
        "Nodes-Average betwee. (Fitting)": "Nodes-BC(Fit)",
        "Nodes-Average eigenv.": "Nodes-EC",
        "Nodes-Average eigenv. (Fitting)": "Nodes-EC(Fit)",
        "Nodes-Average closen.": "Nodes-CC",
        "Nodes-Average closen. (Fitting)": "Nodes-CC(Fit)",
        "Nodes-Assortativity .": "Nodes-ASC",
        "Nodes-Assortativity . (Fitting)": "Nodes-ASC(Fit)",
        "Nodes-Average cluste.": "Nodes-ACC",
        "Nodes-Average cluste. (Fitting)": "Nodes-ACC(Fit)",
        "Nodes-Global efficie.": "Nodes-GE",
        "Nodes-Global efficie. (Fitting)": "Nodes-GE(Fit)",
        "Nodes-Wiener Index": "Nodes-WI",
        "Nodes-Wiener Index (Fitting)": "Nodes-WI(Fit)",
    }

    all_sheets = {}
    for a_file in files:
        if a_file.endswith(allowed_ext):
            # Get the Excel file and load its contents
            file_path = os.path.join(excel_dir_path, a_file)
            file_sheets = pd.read_excel(file_path, sheet_name=None)

            # Append Excel data to one place
            for sheet_name, df in file_sheets.items():
                # Rename it if sheet_name exists in mapping
                new_name = rename_map.get(sheet_name, sheet_name)  # returns the old name if not found in mapping

                # Add the Material column with the file name (without extension)
                df = df.copy()
                mat_label = os.path.splitext(a_file)[0]
                df.insert(0, "Material", mat_label)

                if new_name not in all_sheets:
                    all_sheets[new_name] = []  # initialize list
                all_sheets[new_name].append(df)

    # Concatenate each list of DataFrames into one
    for sheet_name in all_sheets:
        all_sheets[sheet_name] = pd.concat(all_sheets[sheet_name], ignore_index=True)
    return all_sheets


def sgt_csv_to_dataframe(csv_dir_path: str, delimiter: str = ",") -> dict[str, pd.DataFrame] | None:
    """
    Loads multiple CSV files generated by the StructuralGT–Scaling Behavior module into pandas DataFrames.

    This function scans the specified directory for CSV files, reads each one using the given
    delimiter, and stores the results in a dictionary where the keys are file names (without extensions).

    Args:
        csv_dir_path (str): Path to the directory containing CSV files
        delimiter (str, optional): Character used to separate values in the CSV files (default: ",")

    Returns:
        dict[str, pd.DataFrame] | None:
            A dictionary mapping each file name (without extension) to its corresponding
            DataFrame, or None if no valid CSV files are found.
    """

    if csv_dir_path is None:
        return None

    # Get all files in the directory
    files = os.listdir(csv_dir_path)
    files = sorted(files)

    all_sheets = {}
    for a_file in files:
        if a_file.endswith(".csv"):
            # Get the Excel file and load its contents
            csv_path = os.path.join(csv_dir_path, a_file)
            label = os.path.splitext(a_file)[0]   # The file name (without extension)
            df = pd.read_csv(csv_path, delimiter=delimiter)

            if label not in all_sheets:
                all_sheets[label] = df
    return all_sheets


def sgt_spider_plot(df_sgt: pd.DataFrame, labels: list[str], parameters: list[str], value_cols=None) -> None | plt.Figure:
    """
    Generates a spider (radar) plot to compare Graph-Theoretic (GT) parameters 
    across multiple material samples, typically derived from SEM images.

    This visualization helps identify similarities or differences in structural 
    characteristics among materials based on their GT parameter values.

    Args:
        df_sgt (pd.DataFrame): DataFrame containing - 'Material', 'Parameter', and 'value-1', 'value-2', 'value-3', 'value-4' columns
        labels (list[str]): List of material names to include in the comparison
        parameters (list[str]): List of GT parameters to plot along the spider axes
        value_cols (list, optional): List of columns containing GT parameter values. Defaults to [].

    Returns:
        None | matplotlib.figure.Figure:
            The generated Matplotlib Figure if successful, or None if inputs are invalid.
    """

    if value_cols is None:
        value_cols = []

    if df_sgt is None or labels is None or parameters is None:
        return None

    param_rename_map = {
        "Number of nodes": "Nodes",
        "Number of edges": "Edges",
        "Network diameter": "Diameter",
        "Average edge angle (degrees)": "Avg. E. Angle",
        "Median edge angle (degrees)": "Med. E. Angle",
        "Graph density": "GD",
        "Average degree": "AD",
        "Global efficiency": "GE",
        "Wiener Index": "WI",
        "Assortativity coefficient": "ASC",
        "Average clustering coefficient": "ACC",
        "Average betweenness centrality": "BC",
        "Average eigenvector centrality": "EC",
        "Average closeness centrality": "CC",
    }
    if len(value_cols) <= 0:
        value_cols = ["value-1", "value-2", "value-3", "value-4"]

    # Rename Columns: apply replacements in the "Parameter" column
    if "parameter" in df_sgt.columns:
        df_sgt["parameter"] = df_sgt["parameter"].replace(param_rename_map)

    # Ensure the value columns exist
    if all(col in df_sgt.columns for col in value_cols):
        df_sgt["Avg."] = df_sgt[value_cols].astype(float).mean(axis=1)
        df_sgt["Std. Dev."] = df_sgt[value_cols].astype(float).std(axis=1)

    # Filter and pivot
    df_avg = df_sgt.pivot(index='Material', columns='parameter', values='Avg.')
    df_std = df_sgt.pivot(index='Material', columns='parameter', values='Std. Dev.')

    # Ensure consistent parameter order
    df_avg = df_avg[parameters]
    df_std = df_std[parameters]

    # Radar chart setup
    num_vars = len(parameters)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]  # close the loop without mutating the input list

    # Create the figure and axes
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, polar=True)

    # Plot each material
    for mat in labels:
        values = df_avg.loc[mat].tolist()
        values += [values[0]]  # close the loop

        errors = df_std.loc[mat].tolist()
        errors += [errors[0]]

        ax.plot(angles_closed, values, label=mat)
        ax.fill_between(angles_closed,
                        np.array(values) - np.array(errors),
                        np.array(values) + np.array(errors),
                        alpha=0.1)

    # Final touches
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), parameters)
    ax.set_title("Spider Plot with Std. Dev. Error Bands", fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    return fig


def sgt_scaling_plot(y_title: str, df_data: pd.DataFrame, labels: dict, skip_test: bool = False) -> None | plt.Figure:
    """
    Generates a scaling plot showing error bars for a sample material and displays
    corresponding Kolmogorov–Smirnov test results for different statistical fits (Powerlaw, Exponential, Lognormal).
    The right subplot contains only formatted text (no axes, no borders).

    Args:
        y_title (str): Y-axis title
        df_data (pd.DataFrame): DataFrame containing 'Material', 'x-avg', 'y-avg', 'x-std', and 'y-std'
        labels (dict): Mapping of material keys to readable names
        skip_test (bool, optional): Whether to skip the KS test. Defaults to False.

    Returns:
        matplotlib.figure.Figure | None: The generated figure, or None if inputs are invalid.
    """

    if y_title is None or df_data is None or labels is None:
        return None

    # Use pyplot figure so plt.show() works properly
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    ax_1 = fig.add_subplot(2, 2, 1)

    # --- Plot data and compute KS test statistics ---
    fit_text = "Kolmogorov–Smirnov & P-Values\n\n"
    for key, material_name in labels.items():
        df_sample = df_data[df_data['Material'] == key].copy()
        if df_sample.empty:
            continue

        if not skip_test:
            # KS tests for different fits
            res = stats.goodness_of_fit(stats.powerlaw, df_sample['y-avg'].to_numpy())
            res_2 = stats.goodness_of_fit(stats.expon, df_sample['y-avg'].to_numpy())
            res_3 = stats.goodness_of_fit(stats.lognorm, df_sample['y-avg'].to_numpy())

            fit_text += (
                f"{material_name}:\n"
                f"  Power-law → KS={res.statistic:.3f}, p={res.pvalue:.3f}\n"
                f"  Exponential → KS={res_2.statistic:.3f}, p={res_2.pvalue:.3f}\n"
                f"  Log-normal → KS={res_3.statistic:.3f}, p={res_3.pvalue:.3f}\n\n"
            )

        # Error-bar plot
        ax_1.errorbar(
            df_sample['x-avg'],
            df_sample['y-avg'],
            yerr=df_sample['y-std'],
            xerr=df_sample['x-std'],
            label=material_name,
            marker='o',
            capsize=3,
            linestyle='-'
        )

    # --- Format main plot ---
    ax_1.set_xlabel('No. of Nodes', fontsize=12)
    ax_1.set_ylabel(y_title, fontsize=12)
    ax_1.set_title(f'Nodes vs {y_title} (Actual Data)', fontsize=13)
    ax_1.legend(frameon=False)
    ax_1.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)  # cleaner grid
    fig.tight_layout()

    if skip_test:
        fit_text += "Goodness-of-fit tests skipped."

    # --- Create a text-only subplot (no axes, no borders) ---
    ax_2 = fig.add_subplot(2, 2, 2)
    ax_2.axis('off')  # hides axes, ticks, and frame
    ax_2.text(
        0.0, 1.0, fit_text,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='left',
        family='monospace',
        transform=ax_2.transAxes,
        color='black'
    )

    return fig


def upload_to_dropbox(graph_file, folder="/raw_train_data"):
    """
    Uploads graph_file to Dropbox inside App Folder.
    """

    def _load_secrets():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        secrets_path = 'secrets.enc'
        secrets_file = os.path.join(current_dir, secrets_path)
        with open(secrets_file, "rb") as pass_f:
            decrypted = fernet.decrypt(pass_f.read())
            return json.loads(decrypted.decode())

    def _get_access_token(app_key, app_secret, refresh_token):
        """
        Exchanges the refresh token for a short-lived access token.
        """
        token_url = "https://api.dropbox.com/oauth2/token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        auth = (app_key, app_secret)

        response = requests.post(token_url, data=data, auth=auth)
        response.raise_for_status()
        return response.json()["access_token"]

    secrets = _load_secrets()
    access_token = _get_access_token(
        secrets["APP_KEY"],
        secrets["APP_SECRET"],
        secrets["REFRESH_TOKEN"]
    )
    dbx = dropbox.Dropbox(access_token)

    # Ensure the path inside the App Folder
    dest_path = f"{folder}/{os.path.basename(graph_file)}"

    with open(graph_file, "rb") as f:
        dbx.files_upload(
            f.read(),
            dest_path,
            mode=dropbox.files.WriteMode("overwrite")
        )

    return dest_path
