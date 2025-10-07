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
import random
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
from cv2.typing import MatLike
from typing import LiteralString
from dataclasses import dataclass
from cryptography.fernet import Fernet


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
    Read the Excel files (generated by StructuralGT) and save in one DataFrame.

    :param excel_dir_path: Path to the Excel files directory.
    :param allowed_ext: Allowed file extensions (default: ".xlsx").
    :return: A dictionary of DataFrames, where the key is the Excel file name (without extension) and the value is the DataFrame.
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
                new_name = rename_map.get(sheet_name, sheet_name)

                # Add Material column with file name (without extension)
                df = df.copy()
                df["Material"] = os.path.splitext(a_file)[0]

                if new_name not in all_sheets:
                    all_sheets[new_name] = []  # initialize list
                all_sheets[new_name].append(df)

    # Concatenate each list of DataFrames into one
    for sheet_name in all_sheets:
        all_sheets[sheet_name] = pd.concat(all_sheets[sheet_name], ignore_index=True)
    return all_sheets


def gen_spider_plot(df_sgt: pd.DataFrame, materials: list[str], parameters: list[str]) -> None|plt.Figure:
    """
    Creates a spider plot that compares the GT parameters for each material.
    Args:
        df_sgt: A dataframe containing the SGT results.
        materials: A list of material names to include in the plot.
        parameters: A list of GT parameters to be compared.

    Returns: None or a Matplotlib figure.
    """

    if df_sgt is None or materials is None or parameters is None:
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
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each material
    for mat in materials:
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
    plt.tight_layout()
    return fig


def upload_to_dropbox(graph_file, folder="/raw_train_data"):
    """
    Uploads graph_file to Dropbox inside App Folder.
    """

    def _load_secrets():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        secrets_path = 'secrets.enc'
        secrets_file = os.path.join(current_dir, secrets_path)
        with open(secrets_file, "rb") as f:
            decrypted = fernet.decrypt(f.read())
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
