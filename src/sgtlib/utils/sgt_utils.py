# SPDX-License-Identifier: GNU GPL v3

"""
StructuralGT utility functions.
"""

import os
import io
import sys
import cv2
import base64
import logging
import requests
import gsd.hoomd
import subprocess
import scipy as sp
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

@dataclass
class CurveFitModels:
    """
    A collection of common analytic functions used for curve fitting and data modeling.
    Includes the Power-law, Log-normal, Exponential, and Gaussian models.
    """

    @staticmethod
    def run_goodness_of_fit(args):
        """Worker-safe function for parallel execution (returns the serializable result)."""
        name, dist, data = args
        try:
            res = stats.goodness_of_fit(dist, data)
            # Return only serializable data
            return {
                "name": name,
                "ks": float(res.statistic),
                "p": float(res.pvalue),
                "error": None,
            }
        except Exception as e:
            return {"name": name, "ks": np.nan, "p": np.nan, "error": str(e)}

    @staticmethod
    def power_law(x_avg, y_avg, x_fit) -> tuple[np.ndarray, dict] | tuple[None, dict]:
        """
        Fits a power-law model to the given data and returns the fitted curve along with the model parameters.

        The power-law model follows the equation:
            y = a * x^(-k)
        where:
            a → scale (intercept) parameter
            k → decay (exponent) parameter

        Args:
            x_avg (np.ndarray): Array of x-values (independent variable) used for fitting
            y_avg (np.ndarray): Array of y-values (dependent variable) corresponding to x_avg
            x_fit (np.ndarray): Array of x-values over which to generate the fitted curve

        Returns:
            tuple[np.ndarray, dict]:
                - **y_fit** (np.ndarray): The fitted y-values computed from the best-fit parameters over `x_fit`.
                - **params** (dict): A dictionary containing the fitted parameters:
                    {
                        "a": float, # scale parameter
                        "k": float # exponent parameter
                    }

        Notes:
            - Uses `scipy.optimize.curve_fit` to estimate model parameters.
            - Initial parameter guesses are set to [1.0, 1.0].
        """
        def fit_function(x: np.ndarray, a: float, k: float) -> np.ndarray:
            """
            Power-law model: y = a * x^(-k)
            """
            return a * (x ** (-k))

        try:
            init_params = [1.0, 1.0]  # initial guess for [a, k]
            optimal_params: np.ndarray = sp.optimize.curve_fit(fit_function, x_avg, y_avg, p0=init_params)[0]
            a_fit, k_fit = float(optimal_params[0]), float(optimal_params[1])

            # Generate points for the best-fit curve
            y_fit = fit_function(x_fit, a_fit, k_fit)
            return y_fit, {"a": a_fit, "k": k_fit}
        except Exception as err:
            print(err)
            return None, {"a": 0.0, "k": 0.0}

    @staticmethod
    def truncated_power_law(x_avg, y_avg, x_fit) -> tuple[np.ndarray, dict] | tuple[None, dict]:
        """
        Fits a truncated power-law model to the data and returns the fitted curve and parameters.

        The truncated power-law model follows:
            y = a * x^(-k) * exp(-c * x)

        Where:
            - a → scale factor
            - k → exponent of decay
            - c → exponential cutoff parameter

        Args:
            x_avg (np.ndarray): Independent variable values for fitting
            y_avg (np.ndarray): Dependent variable values for fitting
            x_fit (np.ndarray): Points over which to generate the fitted curve

        Returns:
            tuple[np.ndarray, dict]:
                - **y_fit** (np.ndarray): Predicted y-values using best-fit parameters.
                - **params** (dict): {"a": float, "k": float, "c": float}
        """
        def fit_function(x, a, k, c):
            """
            A best-fit model that follows the truncated power law distribution: y = a * x^(-k) * exp(-c * x),
            where a, c, and k are fitting parameters.

            https://en.wikipedia.org/wiki/Power_law#Power_law_with_exponential_cutoff

            Args:
                x (np.array): Array of x values
                a (float): fitting parameter
                k (float): fitting parameter
                c (float): cut-off fitting parameter
            """
            return a * (x ** (-k)) * np.exp(-c * x)

        try:
            init_params_cutoff = [1.0, 1.0, 0.1]
            opt_params_cutoff: np.ndarray = \
                sp.optimize.curve_fit(fit_function, x_avg, y_avg, p0=init_params_cutoff)[0]
            a_fit, k_fit, c_fit = (float(opt_params_cutoff[0]), float(opt_params_cutoff[1]), float(opt_params_cutoff[2]))

            # Generate points for the best-fit curve
            y_fit = fit_function(x_fit, a_fit, k_fit, c_fit)
            return y_fit, {"a": a_fit, "k": k_fit, "c": c_fit}
        except Exception as err:
            print(err)
            return None, {"a": 0.0, "k": 0.0, "c": 0.0}

    @staticmethod
    def lognormal(x_avg, y_avg, x_fit) -> tuple[np.ndarray, dict] | tuple[None, dict]:
        """
        Fits a log-normal model to the data and returns the fitted curve and parameters.

        The log-normal model follows:
            y = a * [1 / (x * σ * sqrt(2π))] * exp(-((ln(x) - μ)²) / (2σ²))

        Where:
            - μ → log-mean
            - σ → log-standard deviation
            - a → amplitude scaling factor

        Args:
            x_avg (np.ndarray): Independent variable values for fitting
            y_avg (np.ndarray): Dependent variable values for fitting
            x_fit (np.ndarray): Points over which to generate the fitted curve

        Returns:
            tuple[np.ndarray, dict]:
                - **y_fit** (np.ndarray): Predicted y-values using best-fit parameters.
                - **params** (dict): {"mu": float, "sigma": float, "a": float}
        """
        def fit_function(x: np.ndarray, mu: float, sigma: float, a: float) -> np.ndarray:
            """
            Log-normal model:
            y = a * [1 / (x * sigma * sqrt(2π))] * exp(-((ln(x) - μ)^2) / (2σ²))
            """
            return a * (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))

        try:
            init_params_log = [0.5, 0.5, 5]  # mu, sigma, a
            opt_params_log: np.ndarray = \
                sp.optimize.curve_fit(fit_function, x_avg, y_avg, p0=init_params_log,
                                      bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]), maxfev=10000)[0]
            mu_fit, sigma_fit, a_fit = float(opt_params_log[0]), float(opt_params_log[1]), float(opt_params_log[2])

            # Generate predicted points for the best-fit curve
            y_fit = fit_function(x_fit, mu_fit, sigma_fit, a_fit)
            return y_fit, {"mu": mu_fit, "sigma": sigma_fit, "a": a_fit}
        except Exception as err:
            print(err)
            return None, {"mu": 0.0, "sigma": 0.0, "a": 0.0}

    @staticmethod
    def exponential(x_avg, y_avg, x_fit) -> tuple[np.ndarray, dict] | tuple[None, dict]:
        """
        Fits an exponential model to the data and returns the fitted curve and parameters.

        The exponential model follows:
            y = a * exp(b * x) + c

        Where:
            - a → amplitude (scale factor)
            - b → growth/decay rate
            - c → vertical offset

        Args:
            x_avg (np.ndarray): Independent variable values for fitting
            y_avg (np.ndarray): Dependent variable values for fitting
            x_fit (np.ndarray): Points over which to generate the fitted curve

        Returns:
            tuple[np.ndarray, dict]:
                - **y_fit** (np.ndarray): Predicted y-values using best-fit parameters.
                - **params** (dict): {"a": float, "b": float, "c": float}
        """
        def fit_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a * np.exp(b * x) + c

        try:
            init_params = [1.0, -0.1, 0.0]
            opt_params: np.ndarray = sp.optimize.curve_fit(
                fit_function, x_avg, y_avg, p0=init_params, maxfev=2000
            )[0]

            a_fit, b_fit, c_fit = map(float, opt_params)
            y_fit = fit_function(x_fit, a_fit, b_fit, c_fit)
            return y_fit, {"a": a_fit, "b": b_fit, "c": c_fit}
        except Exception as err:
            print(err)
            return None, {"a": 0.0, "b": 0.0, "c": 0.0}

    @staticmethod
    def linear(x_avg: np.ndarray, y_avg: np.ndarray, x_fit: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Fits a linear (first-degree polynomial) model to the data.

        Model:
            y = m * x + b

        where:
            m → slope of the line
            b → intercept

        This model corresponds to a standard linear regression
        and is useful for approximating monotonic relationships between variables.

        Args:
            x_avg (np.ndarray): Independent variable values for fitting
            y_avg (np.ndarray): Dependent variable values for fitting
            x_fit (np.ndarray): Points over which to generate the fitted line

        Returns:
            tuple[np.ndarray, dict]:
                - y_fit (np.ndarray): Predicted y-values using the best-fit line
                - params (dict): {"m": float, "b": float}
        """

        def fit_function(x, m, b):
            return m * x + b

        init_params = [1.0, 0.0]
        opt_params = sp.optimize.curve_fit(fit_function, x_avg, y_avg, p0=init_params, maxfev=1000)[0]
        m_fit, b_fit = map(float, opt_params)
        y_fit = fit_function(x_fit, m_fit, b_fit)
        return y_fit, {"m": m_fit, "b": b_fit}

    @staticmethod
    def gaussian(x_avg, y_avg, x_fit) -> tuple[np.ndarray, dict] | tuple[None, dict]:
        """
        Fits a Gaussian (normal) distribution model to the data and returns the fitted curve and parameters.

        The Gaussian model follows:
            y = a * exp(-((x - μ)²) / (2σ²))

        Where:
            - μ → mean (center of the peak)
            - σ → standard deviation (controls spread)
            - a → amplitude (peak height)

        Args:
            x_avg (np.ndarray): Independent variable values for fitting
            y_avg (np.ndarray): Dependent variable values for fitting
            x_fit (np.ndarray): Points over which to generate the fitted curve

        Returns:
            tuple[np.ndarray, dict]:
                - **y_fit** (np.ndarray): Predicted y-values using best-fit parameters.
                - **params** (dict): {"mu": float, "sigma": float, "a": float}
        """
        def fit_function(x: np.ndarray, mu: float, sigma: float, a: float) -> np.ndarray:
            return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        try:
            init_params = [np.mean(x_avg), np.std(x_avg), max(y_avg)]
            opt_params: np.ndarray = sp.optimize.curve_fit(
                fit_function, x_avg, y_avg, p0=init_params, maxfev=2000
            )[0]

            mu_fit, sigma_fit, a_fit = map(float, opt_params)
            y_fit = fit_function(x_fit, mu_fit, sigma_fit, a_fit)
            return y_fit, {"mu": mu_fit, "sigma": sigma_fit, "a": a_fit}
        except Exception as err:
            print(err)
            return None, {"mu": 0.0, "sigma": 0.0, "a": 0.0}

    @staticmethod
    def gamma(x_avg: np.ndarray, y_avg: np.ndarray, x_fit: np.ndarray) -> tuple[np.ndarray, dict] | tuple[None, dict]:
        """
        Fits a Gamma distribution model to the given (x_avg, y_avg) data.

        The Gamma probability density function (PDF) is defined as:
            y = a * [x^(k-1) * exp(-x/θ)] / [θ^k * Γ(k)]

        Where:
            - a is a scaling factor,
            - k (shape) and θ (scale) are the distribution parameters,
            - Γ(k) is the Gamma function.

        This model is useful for positively skewed data, commonly appearing in
        lifetime or waiting-time distributions.

        Args:
            x_avg (np.ndarray): Independent variable values
            y_avg (np.ndarray): Dependent variable values (to fit)
            x_fit (np.ndarray): Points at which to generate the fitted curve

        Returns:
            tuple[np.ndarray, dict]:
                y_fit (np.ndarray): Best-fit curve values
                params (dict): Dictionary containing fitted parameters {a, k, theta}
        """

        def fit_function(x: np.ndarray, a: float, alpha: float, theta: float) -> np.ndarray:
            """
            Gamma model:
            y = a * [x^(k-1) * exp(-x/θ)] / [θ^k * Γ(k)]
            """
            gamma_k = sp.special.gamma(alpha)
            return a * ((x ** (alpha - 1)) * np.exp(-x / theta)) / ((theta ** alpha) * gamma_k)

        try:
            # Initial guesses for [a, alpha, theta]
            init_params_gamma = [0, 1.0, 2.0]

            # Fit the curve to data
            opt_params_gamma: np.ndarray = sp.optimize.curve_fit(
                fit_function,
                x_avg,
                y_avg,
                p0=init_params_gamma,
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                maxfev=1000
            )[0]

            a_fit, alpha_fit, theta_fit = float(opt_params_gamma[0]), float(opt_params_gamma[1]), float(opt_params_gamma[2])

            # Generate predicted points for the best-fit curve
            y_fit = fit_function(x_fit, a_fit, alpha_fit, theta_fit)
            return y_fit, {"a": a_fit, "alpha": alpha_fit, "theta": theta_fit}
        except Exception as err:
            print(err)
            return None, {"a": 0.0, "k": 0.0, "theta": 0.0}


@dataclass
class MaximumLikelihoodEstimation:
    """"""

    @staticmethod
    def lognormal(x_avg: np.ndarray, y_avg: np.ndarray):
        """"""

        # Define log-likelihood
        def neg_loglike(params, norm_data: np.ndarray, x_data: np.ndarray):
            """Negative log-likelihood function for the log-normal distribution."""
            alpha_0, alpha_1, beta_0, beta_1 = params
            mu = alpha_0 + alpha_1 * np.log(norm_data)
            sigma = np.clip(beta_0 + beta_1 * np.log(norm_data), 1e-6, None)
            ll = np.sum(sp.stats.norm.logpdf(np.log(x_data), loc=mu, scale=sigma) - np.log(x_data))
            return -ll  # negative for minimization

        # Fit via MLE
        init = np.array([-1, 0.5, 0.3, 0.05])
        res = sp.optimize.minimize(neg_loglike, init, args=(x_avg, y_avg))
        a0, a1, b0, b1 = res.x
        print("Fitted parameters:")
        print(f"alpha0={a0:.3f}, alpha1={a1:.3f}, beta0={b0:.3f}, beta1={b1:.3f}")
        return {"alpha0": a0, "alpha1": a1, "beta0": b0, "beta1": b1}


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
    import socket
    import platform
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


def sgt_spider_plot(df_sgt: pd.DataFrame, labels: dict, parameters: list[str], value_cols=None) -> None | plt.Figure:
    """
    Generates a spider (radar) plot to compare Graph-Theoretic (GT) parameters 
    across multiple material samples, typically derived from SEM images.

    This visualization helps identify similarities or differences in structural 
    characteristics among materials based on their GT parameter values.

    Args:
        df_sgt (pd.DataFrame): DataFrame containing - 'Material', 'Parameter', and 'value-1', 'value-2', 'value-3', 'value-4' columns
        labels (dict): Mapping of material keys to readable names
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
        df_sgt["Avg."] = df_sgt[value_cols].to_numpy().mean(axis=1)
        df_sgt["Std. Dev."] = df_sgt[value_cols].to_numpy().std(axis=1)

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
    ax = fig.add_subplot(1, 1, 1, projection='polar')

    # Plot each material
    for key, material_name in labels.items():
        values = df_avg.loc[key].tolist()
        values += [values[0]]  # close the loop

        errors = df_std.loc[key].tolist()
        errors += [errors[0]]

        ax.plot(angles_closed, values, label=material_name)
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


def sgt_scaling_plot(y_title: str, df_data: pd.DataFrame, labels: dict, skip_test: bool = False, fit_func: str = None) -> None | plt.Figure:
    """
    Generates a scaling plot showing error bars for a sample material and displays
    corresponding Kolmogorov–Smirnov test results for different statistical fits (Powerlaw, Exponential, Lognormal).
    The right subplot contains only formatted text (no axes, no borders).

    Args:
        y_title (str): Y-axis title
        df_data (pd.DataFrame): DataFrame containing 'Material', 'x-avg', 'y-avg', 'x-std', and 'y-std'
        labels (dict): Mapping of material keys to readable names
        skip_test (bool, optional): Whether to skip the KS test. Defaults to False
        fit_func (str, optional): Function to fit the data (log-normal, power-law, exponential). Defaults to None

    Returns:
        matplotlib.figure.Figure | None: The generated figure, or None if inputs are invalid.
    """

    def parallel_goodness_of_fit(df_distribution, sample_name):
        """
        Run multiple goodness-of-fit tests (KS test) in parallel
        for several candidate distributions.

        Args:
            df_distribution: pandas DataFrame containing the sample data (column 'y-avg')
            sample_name: str, name of the material for output labeling

        Returns:
            str: formatted text summary of KS and p-values for each distribution
        """
        data = df_distribution['y-avg'].to_numpy()

        # Define distributions to test
        distributions = {
            #"Power Law": stats.powerlaw,
            #"Exponential": stats.expon,
            #"Log Normal": stats.lognorm,
            #"Gamma": stats.gamma,
            "Weibull": stats.weibull_min,
            "Inverse Gaussian": stats.wald,
            "Generalized Pareto": stats.genpareto
        }

        # Prepare arguments for each parallel process
        args_list = [(name, dist, data) for name, dist in distributions.items()]

        # Use multiprocessing pool
        with mp.Pool(processes=min(len(distributions), mp.cpu_count())) as pool:
            results = pool.map(CurveFitModels.run_goodness_of_fit, args_list)

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)

        # Format readable summary text
        fmt_text = f"{sample_name}:\n"
        for _, row in df_results.iterrows():
            if row["error"]:
                fmt_text += f"  {row['name']} → ERROR\n"
            else:
                fmt_text += f"  {row['name']} → KS={row['ks']:.3f}, p={row['p']:.3f}\n"
        return fmt_text

    if y_title is None or df_data is None or labels is None:
        return None

    if y_title == "Kernel Size" and 'kernel-dim' in df_data.columns:
        df_data = df_data.copy()
        df_data['y-avg'] = df_data['kernel-dim']
        df_data['y-std'] = 0.0

    # Use pyplot figure so plt.show() works properly
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    # Main 2x2 grid
    gs = fig.add_gridspec(2, 2)
    ax_1 = fig.add_subplot(gs[0, 0])         # Actual data with error bars
    ax_2 = fig.add_subplot(gs[0, 1])         # goodness-of-fit test results
    ax_3 = None
    ax_4 = None
    ax_4_grids, i = [], 0
    if type(fit_func) == str:
        ax_3 = fig.add_subplot(gs[1, 0])     # Curve fits with selected distributions
        ax_4 = fig.add_subplot(gs[1, 1])

        # Subdivide the (1,1) slot (ax_4 area) into a 2x2 grid
        gs_sub = gs[1, 1].subgridspec(2, 2)
        ax_4_1 = fig.add_subplot(gs_sub[0, 0])
        ax_4_2 = fig.add_subplot(gs_sub[0, 1])
        ax_4_3 = fig.add_subplot(gs_sub[1, 0])
        ax_4_4 = fig.add_subplot(gs_sub[1, 1])
        ax_4_grids = [ax_4_1, ax_4_2, ax_4_3, ax_4_4]

    # --- Plot data and compute KS test statistics ---
    txt_test = "Kolmogorov–Smirnov & P-Values\n\n"
    for key, material_name in labels.items():
        df_sample = df_data[df_data['Material'] == key].copy()
        if df_sample.empty:
            continue

        # Perform the Goodness-of-fit test?
        if not skip_test:
            # KS tests for different fits
            txt_test += parallel_goodness_of_fit(df_sample, material_name)

        # Plot Curves fitted to specific distributions
        if ax_3 is not None:
            x_avg = df_sample['x-avg'].to_numpy()
            y_avg = df_sample['y-avg'].to_numpy()
            x_fit = np.linspace(min(x_avg), max(max(x_avg), 10000), 100)
            y_fit, axis_label = None, ""
            if fit_func == "lognorm":
                y_fit, params = CurveFitModels.lognormal(x_avg, y_avg, x_fit)
                mu_fit, sigma_fit, a_log_fit = params["mu"], params["sigma"], params["a"]
                axis_label = f'{material_name}: a={a_log_fit:.2f}, $\\mu={mu_fit:.3f}$, $\\sigma={sigma_fit:.3f}$'

                # Fit log-normal distribution to y_avg
                shape, loc, scale = stats.lognorm.fit(y_avg, floc=0)  # floc=0 fixes location at 0 (common for lognorm)
                # Generate theoretical quantiles for the QQ plot, we compare sorted empirical y vs. theoretical quantiles
                quantiles = np.linspace(0.01, 0.99, len(y_avg))
                theoretical_q = stats.lognorm.ppf(quantiles, shape, loc=loc, scale=scale)
                empirical_q = np.quantile(y_avg, quantiles)
                # Plot QQ-plot
                if i < len(ax_4_grids):
                    ax_4_grids[i].plot(theoretical_q, theoretical_q, 'r--', label="Identity Line")
                    ax_4_grids[i].scatter(theoretical_q, empirical_q, alpha=0.7, edgecolor="k", linewidths=0.5, s=6, label=f"{material_name}")
                    if i in (0, 2):
                        ax_4_grids[i].set_ylabel("Empirical Quantiles", fontsize=6)
                    ax_4_grids[i].set_xlabel("Theoretical Quantiles (Lognormal)", fontsize=6)
                    ax_4_grids[i].tick_params(labelsize=5)
                    ax_4_grids[i].legend(fontsize=6)
                    ax_4_grids[i].set_frame_on(True)  # keep only small subplot borders visible
                    ax_4_grids[i].grid(linestyle="--", linewidth=0.5, alpha=0.25)
                    i += 1
            elif fit_func == "powerlaw":
                y_fit, params = CurveFitModels.power_law(x_avg, y_avg, x_fit)
                a_fit, k_fit = params["a"], params["k"]
                axis_label = f'{material_name}: $a={a_fit:.3f}, k={k_fit:.3f}$'
            elif fit_func == "linear":
                y_fit, params = CurveFitModels.linear(x_avg, y_avg, x_fit)
                slope_fit, intercept_fit = params["m"], params["b"]
                axis_label = f'{material_name}: $slope={slope_fit:.3f}, b={intercept_fit:.3f}$'
            elif fit_func == "gamma":
                y_fit, params = CurveFitModels.gamma(x_avg, y_avg, x_fit)
                a_fit, alpha, theta = params["a"], params["alpha"], params["theta"]
                axis_label = f'{material_name}: a={a_fit:.3f}, $\\alpha={alpha:.3f}$, $\\theta={theta:.3f}$'
            ax_3.plot(x_fit, y_fit, label=axis_label, linestyle='-') if y_fit is not None else None

        # Plot the best scale with an 'x' symbol
        legend_label = None
        if y_title == "Kernel Size":
            # --- Copy last row as dict ---
            last_row_dict = df_sample.iloc[-1].to_dict()
            # --- Delete last row ---
            df_sample = df_sample.iloc[:-1].copy()
            ax_1.scatter(last_row_dict['x-avg'], last_row_dict['y-avg'], marker='x')
            # Add Horizontal Line
            ax_1.axhline(
                y=last_row_dict['y-avg'],
                linestyle='--',
                linewidth=0.2,
                # label=f"y = {last_row_dict['y-avg']:.2f}"
            )
            legend_label = f"{material_name} (y={last_row_dict['y-avg']:.2f}px)"

        # Error-bar plot
        ax_1.errorbar(
            df_sample['x-avg'],
            df_sample['y-avg'],
            yerr=df_sample['y-std'],
            xerr=df_sample['x-std'],
            label=material_name if legend_label is None else legend_label,
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

    if skip_test:
        txt_test += "Goodness-of-fit tests skipped."

    # --- Create a text-only subplot (no axes, no borders) ---
    ax_2.axis('off')  # hides axes, ticks, and frame
    ax_2.text(
        0.0, 1.0, txt_test,
        fontsize=8,
        verticalalignment='top',
        horizontalalignment='left',
        family='monospace',
        transform=ax_2.transAxes,
        color='black'
    )

    # --- Draw curve fits using selected distributions (power-law or log-normal or exponential)
    if ax_3 is not None:
        if fit_func == "lognorm":
            ax_3.set_title(
                r"LogNormal Fit: $y = a \cdot \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln{x}-\mu)^2}{2\sigma^2}}$"
                f"\nNodes vs {y_title}",
                fontsize=10
            )
            ax_4.set_title(f"Q–Q Plot: Lognormal Fit for {y_title}")
        elif fit_func == "powerlaw":
            ax_3.set_title(
                r"PowerLaw Fit: $y = a x^{-k}$"
                f"\nNodes vs {y_title}",
                fontsize=10
            )
        elif fit_func == "linear":
            ax_3.set_title(
                r"Linear Fit: $y = m(x) + b$"
                f"\nNodes vs {y_title}",
                fontsize=10
            )
        elif fit_func == "gamma":
            ax_3.set_title(
                r"Gamma Fit: $y = a \cdot x^{-k} \cdot \exp\left(-\frac{x}{a}\right)$"
                f"\nNodes vs {y_title}",
                fontsize=10
            )
        else:
            fig.tight_layout()
            return fig
        ax_3.set_xlabel('No. of Nodes', fontsize=12)
        ax_3.set_ylabel(y_title, fontsize=12)
        ax_3.legend(frameon=False)
        ax_3.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)  # cleaner grid

        ax_4.axis("off")  # hide all ticks and labels
        ax_4.set_frame_on(False)  # remove the border/frame

    fig.tight_layout()
    return fig


def upload_to_dropbox(graph_file, folder="/raw_train_data"):
    """
    Uploads graph_file to Dropbox inside the App Folder.
    """
    import json
    import dropbox
    from cryptography.fernet import Fernet

    def _load_secrets():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        secrets_path = 'secrets.enc'
        secrets_file = os.path.join(current_dir, secrets_path)
        with open(secrets_file, "rb") as pass_f:
            fernet = Fernet()
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
            mode=dropbox.files.WriteMode.overwrite
        )

    return dest_path
