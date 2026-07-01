# SPDX-License-Identifier: GNU GPL v3

"""
StructuralGT module for generating plots and DataFrames
"""

import os
import scipy as sp
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass


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
            print(e)
            return {"name": name, "ks": np.nan, "p": np.nan, "error": str(e)}

    @staticmethod
    def power_law(x_avg: np.ndarray|None, y_avg: np.ndarray|None, x_fit: np.ndarray|None) -> tuple[np.ndarray, dict] | tuple[None, dict]:
        """
        Fits a power-law model to the given data and returns the fitted curve along with the model parameters.

        The power-law model follows the equation:
            y = a * x^(-k)
        where:
            a → amplitude -- scale (intercept) parameter
            k → decay (exponent) parameter

        Args:
            x_avg (np.ndarray): Array of x-values (independent variable) used for fitting
            y_avg (np.ndarray): Array of y-values (dependent variable) corresponding to x_avg
            x_fit (np.ndarray): Array of x-values over which to generate the fitted curve

        Returns:
            tuple[np.ndarray, dict]:
                - **y_fit** (np.ndarray): The fitted y-values computed from the best-fit parameters over `x_fit`.
                - **params** (dict): A dictionary containing the fitted parameters:
                    {"a": float, "k": float # exponent parameter}

        Notes:
            - Uses `scipy.optimize.curve_fit` to estimate model parameters.
            - Initial parameter guesses are set to [1.0, 1.0].
        """
        if x_avg is None or y_avg is None or x_fit is None:
            return None, {"a": 0.0, "k": 0.0}

        def fit_function(x: np.ndarray, a: float, k: float) -> np.ndarray:
            """
            Power-law model: y = a * x^(-k)
            """
            return a * (x ** (-k))

        try:
            init_params = [1.0, 1.0]  # initial guess for [a, k]
            optimal_params: np.ndarray = sp.optimize.curve_fit(fit_function, x_avg, y_avg, p0=init_params,
                                                            bounds = ([1e-10, 0.01], [np.inf, 10.0]), maxfev=10000)[0]
            a_fit, k_fit = float(optimal_params[0]), float(optimal_params[1])

            # Generate points for the best-fit curve
            y_fit = fit_function(x_fit, a_fit, k_fit)
            return y_fit, {"a": a_fit, "k": k_fit}
        except Exception as err:
            print(err)
            return None, {"a": 0.0, "k": 0.0}

    @staticmethod
    def stretched_power_law(x_avg: np.ndarray|None, y_avg: np.ndarray|None, x_fit: np.ndarray|None) -> tuple[np.ndarray, dict] | tuple[None, dict]:
        """
        Fits a stretched power-law (Weibull-tail power-law) model to the given data and returns
        the fitted curve along with the model parameters.

        The stretched power-law model follows the equation:
            y = a * x^(-k) * exp(-(x / x_c)^beta)
        where:
            a → amplitude (scale) parameter
            k → power-law exponent
            x_c → cutoff scale parameter
            beta → stretching exponent (controls tail shape)

        Args:
            x_avg (np.ndarray): Array of x-values (independent variable) used for fitting
            y_avg (np.ndarray): Array of y-values (dependent variable) corresponding to x_avg
            x_fit (np.ndarray): Array of x-values over which to generate the fitted curve

        Returns:
            tuple[np.ndarray, dict]:
                - **y_fit** (np.ndarray): The fitted y-values computed from the best-fit parameters over `x_fit`.
                - **params** (dict): A dictionary containing the fitted parameters:
                    {"a": float, "k": float, "x_c": float, "beta": float}

        Notes:
            - Uses `scipy.optimize.curve_fit` to estimate model parameters.
            - Initial parameter guesses are set to [1.0, 1.0, 1.0, 1.0].
        """

        if x_avg is None or y_avg is None or x_fit is None:
            return None, {"a": 0.0, "k": 0.0, "x_c": 0.0, "beta": 0.0}

        def fit_function(x: np.ndarray, a: float, k: float, x_c: float, beta: float) -> np.ndarray:
            """
            Stretched power-law model: y = a * x^(-k) * exp(-(x / x_c)^beta)
            """
            return a * (x ** (-k)) * np.exp(-(x / x_c) ** beta)

        try:
            init_params = [0.01, 0.01, 1.0, 0.1]  # initial guesses for [a, k, x_c, beta]
            optimal_params: np.ndarray = sp.optimize.curve_fit(fit_function, x_avg, y_avg, p0=init_params,
                                    bounds = ([1e-10, 0.01, 1e-10, 0.01],[np.inf, 10.0, np.inf, 5.0]), maxfev=10000)[0]
            a_fit, k_fit, x_c_fit, beta_fit = (float(optimal_params[0]), float(optimal_params[1]),
                                               float(optimal_params[2]), float(optimal_params[3]))

            # Generate points for the best-fit curve
            y_fit = fit_function(x_fit, a_fit, k_fit, x_c_fit, beta_fit)
            return y_fit, {"a": a_fit, "k": k_fit, "x_c": x_c_fit, "beta": beta_fit}
        except Exception as err:
            print("Stretched power-law fit error:", err)
            return None, {"a": 0.0, "k": 0.0, "x_c": 0.0, "beta": 0.0}

    @staticmethod
    def lognormal(x_avg: np.ndarray|None, y_avg: np.ndarray|None, x_fit: np.ndarray|None) -> tuple[np.ndarray, dict] | tuple[None, dict]:
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

        if x_avg is None or y_avg is None or x_fit is None:
            return None, {"mu": 0.0, "sigma": 0.0, "a": 0.0}

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
    def exponential(x_avg: np.ndarray|None, y_avg: np.ndarray|None, x_fit: np.ndarray|None) -> tuple[np.ndarray, dict] | tuple[None, dict]:
        """
        Fits an exponential model to the data and returns the fitted curve and parameters.

        The exponential model follows:
            y = a * exp(λ * x) + c

        Where:
            - a → amplitude (scale factor)
            - λ → growth/decay rate
            - c → vertical offset

        Args:
            x_avg (np.ndarray): Independent variable values for fitting
            y_avg (np.ndarray): Dependent variable values for fitting
            x_fit (np.ndarray): Points over which to generate the fitted curve

        Returns:
            tuple[np.ndarray, dict]:
                - **y_fit** (np.ndarray): Predicted y-values using best-fit parameters.
                - **params** (dict): {"a": float, "lambda": float, "c": float}
        """

        if x_avg is None or y_avg is None or x_fit is None:
            return None, {"a": 0.0, "lambda": 0.0, "c": 0.0}

        def fit_function(x: np.ndarray, a: float, lamda: float, c: float) -> np.ndarray:
            return a * np.exp(lamda * x) + c

        try:
            init_params = [1.0, 0.1, 0.0]
            opt_params: np.ndarray = sp.optimize.curve_fit(
                fit_function, x_avg, y_avg, p0=init_params,
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]), maxfev=5000 )[0]

            a_fit, b_fit, c_fit = map(float, opt_params)
            y_fit = fit_function(x_fit, a_fit, b_fit, c_fit)
            return y_fit, {"a": a_fit, "lambda": b_fit, "c": c_fit}
        except Exception as err:
            print(err)
            return None, {"a": 0.0, "lambda": 0.0, "c": 0.0}

    @staticmethod
    def gamma(x_avg: np.ndarray|None, y_avg: np.ndarray|None, x_fit: np.ndarray|None) -> tuple[np.ndarray, dict] | tuple[None, dict]:
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

        if x_avg is None or y_avg is None or x_fit is None:
            return None, {"a": 0.0, "k": 0.0, "theta": 0.0}

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

            a_fit, alpha_fit, theta_fit = float(opt_params_gamma[0]), float(opt_params_gamma[1]), float(
                opt_params_gamma[2])

            # Generate predicted points for the best-fit curve
            y_fit = fit_function(x_fit, a_fit, alpha_fit, theta_fit)
            return y_fit, {"a": a_fit, "alpha": alpha_fit, "theta": theta_fit}
        except Exception as err:
            print(err)
            return None, {"a": 0.0, "k": 0.0, "theta": 0.0}

    @staticmethod
    def linear(x_avg: np.ndarray|None, y_avg: np.ndarray|None, x_fit: np.ndarray|None) -> tuple[np.ndarray, dict] | tuple[None, dict]:
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

        if x_avg is None or y_avg is None or x_fit is None:
            return None, {"m": 0.0, "b": 0.0}

        def fit_function(x, m, b):
            return m * x + b

        init_params = [1.0, 0.0]
        opt_params = sp.optimize.curve_fit(fit_function, x_avg, y_avg, p0=init_params, maxfev=1000)[0]
        m_fit, b_fit = map(float, opt_params)
        y_fit = fit_function(x_fit, m_fit, b_fit)
        return y_fit, {"m": m_fit, "b": b_fit}


@dataclass
class QQPlots:
    """"""

    @staticmethod
    def qq_plot(empirical_q: float|np.ndarray, theoretical_q: float|np.ndarray, ax: plt.Axes, legend_txt: str,
                use_log_scale: bool = True,
                upper_band: float|np.ndarray|None = None,
                lower_band: float|np.ndarray|None = None):
        """"""
        # 1. 95% confidence band (approximate)
        # For n > 30, the 95% confidence envelope around the line y=x can be approximated as: ± 1.36 / sqrt(n)
        n = len(empirical_q) if isinstance(empirical_q, np.ndarray) else 1
        band = 1.36 / np.sqrt(n) if n > 0 else 0.0
        upper_band = theoretical_q + band * theoretical_q if upper_band is None else upper_band
        lower_band = theoretical_q - band * theoretical_q if lower_band is None else lower_band

        # 2. QQ Plot
        ax.plot(theoretical_q, theoretical_q, 'r--', label="Identity Line")
        ax.fill_between(theoretical_q, lower_band, upper_band, color="gray", alpha=0.2, label="95% Confidence Band")
        ax.scatter(theoretical_q, empirical_q, alpha=0.7, edgecolor="k", linewidths=0.5, s=6, label=f"{legend_txt}")
        if use_log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=6)
        ax.grid(linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_frame_on(True)  # keep only small subplot borders visible

    @staticmethod
    def log_qq_plot(distribution_data: np.ndarray, ax: plt.Axes, legend_txt: str, show_y_label: bool = False):
        """
        Direct (unconditional) Q-Q plot comparing our samples to a log-normal distribution (or any other distribution).
        Direct comparison should tell us: "do the observed values distribution_data approximately follow a log-normal
        distribution?"

        This approach is scale-sensitive -- a small mismatch in shape or scale can make the plot look bad even if
        underlying transformed data are nearly normal (or match specified distribution). To help with this, we generate
         a Log-transformed QQ-Plot.

        Returns:

        """
        # 1. Clean and sort
        data = np.asarray(distribution_data)
        data = data[data > 0]
        data = np.sort(data)

        # 2. Fit Lognormal parameters (mu, sigma) via MLE on log(data)
        log_data = np.log(data)
        mu, sigma = np.mean(log_data), np.std(log_data, ddof=1) # OR mu, sigma = sp.stats.norm.fit(log_data)

        # 3. Compute theoretical quantiles from Lognormal inverse CDF
        quantiles = np.linspace(0.01, 0.99, len(data))
        theoretical_q = stats.lognorm.ppf(quantiles, s=sigma, scale=np.exp(mu))
        empirical_q = np.quantile(data, quantiles)

        # 4. QQ Plot for log-transformed data
        QQPlots.qq_plot(empirical_q, theoretical_q, ax, legend_txt, use_log_scale=True)
        if show_y_label:
            ax.set_ylabel("Empirical Quantiles (log(data))", fontsize=6)
        ax.set_xlabel("Theoretical Quantiles (Normal)", fontsize=6)

        return "Log Q–Q Plot (Lognormal)"

    @staticmethod
    def gamma_qq_plot(distribution_data: np.ndarray, ax: plt.Axes, legend_txt: str, show_y_label: bool = False):
        """
        Direct (unconditional) Q–Q plot comparing samples to a Gamma distribution.

        This method evaluates whether the observed data approximately follow
        a Gamma distribution by comparing empirical quantiles to those derived
        from the fitted Gamma model.

        The Gamma distribution has PDF:
            f(x; k, θ) = x^(k-1) * exp(-x/θ) / [Γ(k) * θ^k]
        where:
            k → shape parameter
            θ → scale parameter

        Steps:
          1. Fit Gamma distribution parameters using MLE.
          2. Compute theoretical quantiles from Gamma inverse CDF (ppf).
          3. Compare to empirical quantiles from the data.

        Returns:
            str: Plot title identifier.
        """

        # 1. Clean and sort
        data = np.asarray(distribution_data)
        data = data[data > 0]  # Gamma is defined only for positive values
        data = np.sort(data)

        # 2. Fit Gamma parameters (shape=k, loc, scale=θ) via MLE
        shape_fit, loc_fit, scale_fit = stats.gamma.fit(data, floc=0)  # force loc=0 for stability

        # 3. Compute theoretical quantiles from fitted Gamma inverse CDF
        quantiles = np.linspace(0.01, 0.99, len(data))
        theoretical_q = stats.gamma.ppf(quantiles, a=shape_fit, loc=loc_fit, scale=scale_fit)
        empirical_q = np.quantile(data, quantiles)

        # 4. Generate the Q–Q plot
        QQPlots.qq_plot(empirical_q, theoretical_q, ax, legend_txt, use_log_scale=True)
        if show_y_label:
            ax.set_ylabel("Empirical Quantiles (data)", fontsize=6)
        ax.set_xlabel("Theoretical Quantiles (Gamma)", fontsize=6)

        return "Log Q–Q Plot (Gamma)"

    @staticmethod
    def pwr_qq_plot(distribution_data: np.ndarray, ax: plt.Axes, legend_txt: str, show_y_label: bool = False):
        """
        Unconditional Q–Q plot for a 1D sample against a power-law distribution. We are not fitting a model y = a * x^{-k},
        but instead testing if the empirical data (x) follow a power-law probability distribution:

        p(x) ∝ x^{−α}, x >= x_min

        Fits parameters (alpha, x_min) to x, computes theoretical quantiles
        from the fitted power-law, and compares them to empirical quantiles.

        Args:
            distribution_data (np.ndarray): Sample data (x > 0)
            ax (plt.Axes): Matplotlib Axes for plotting
            legend_txt (str): Label for the dataset
            show_y_label (bool): Whether to display Y-axis label
        """
        # 1. Ensure strictly positive data
        data = np.asarray(distribution_data)
        data = data[data > 0]
        data = np.sort(data)

        # 2b. Fit power-law parameters using MLE: alpha_hat = 1 + n / sum(log(y/x_min))
        x_min = data.min()  # assuming x_min is the smallest value in the sample
        alpha_hat = 1 + len(data) / np.sum(np.log(data / x_min))

        # 3. Compute theoretical quantiles from power-law CDF --- Inverse CDF (quantile function):  Q(p) = x_min * (1 - p)^(-1/(alpha - 1))
        quantiles = np.linspace(0.01, 0.99, len(data))
        theoretical_q = x_min * (1 - quantiles) ** (-1 / (alpha_hat - 1))
        empirical_q = np.quantile(data, quantiles)

        # 4. QQ Plot on Log-Log scale
        QQPlots.qq_plot(empirical_q, theoretical_q, ax, legend_txt, use_log_scale=True)
        if show_y_label:
            ax.set_ylabel("Empirical Quantiles (data)", fontsize=6)
        ax.set_xlabel("Theoretical Quantiles (Power Law)", fontsize=6)

        return "Log Q–Q Plot (Power Law)"

    @staticmethod
    def stretched_pwr_qq_plot(distribution_data: np.ndarray, ax: plt.Axes, legend_txt: str, show_y_label: bool = False):
        """
        Unconditional Q–Q Plot for Power-Law with Stretched-Exponential Cutoff distribution. We are testing if the empirical data (x)
        follow a stretched power-law probability distribution. It is unconditional because we do not have y for fitting
        y = a * x^(-k) * exp(-(x / x_c)^beta).

        Instead, we use PDF:
            p(x) ∝ x^(-alpha) * exp(-lambda * x), for x >= x_min

        This plot visually compares the empirical quantiles of the sample to
        the theoretical quantiles from the fitted model. If points lie roughly
        along the identity line (y = x), the sample follows this distribution.

        Args:
            distribution_data (np.ndarray): Sample data (must be positive)
            ax (plt.Axes): Axis on which to draw the QQ plot
            legend_txt (str): Legend label (e.g., material or dataset name)
            show_y_label (bool): Whether to show the Y-axis label

        Returns:
            str: Description of the plot.
        """

        def stretched_powerlaw_pdf(x, alpha, lamb_da, x_min):
            """PDF: p(x) ∝ x^(-alpha) * exp(-lambda * x), for x >= x_min"""
            norm_const, _ = sp.integrate.quad(lambda t: t ** (-alpha) * np.exp(-lamb_da * t), x_min, np.inf)
            return (x ** (-alpha)) * np.exp(-lamb_da * x) / norm_const

        def stretched_powerlaw_cdf(x, alpha, lamb_da, x_min):
            """Numerically integrate PDF to get CDF"""
            norm_const, _ = sp.integrate.quad(lambda t: t ** (-alpha) * np.exp(-lamb_da * t), x_min, np.inf)
            num, _ = sp.integrate.quad(lambda t: t ** (-alpha) * np.exp(-lamb_da * t), x_min, x)
            return num / norm_const

        def fit_stretched_powerlaw(x, x_min=None):
            """Fit stretched power-law via MLE (approximate for continuous data)"""
            x = x[x > 0]
            if x_min is None:
                x_min = x.min()

            def neg_log_likelihood(params) -> float:
                alpha, lamb_da = params
                if alpha <= 1 or lamb_da <= 0:
                    return np.inf
                pdf_vals = stretched_powerlaw_pdf(x, alpha, lamb_da, x_min)
                pdf_vals: np.ndarray = np.clip(pdf_vals, 1e-12, None)
                return np.sum(np.log(pdf_vals)) * -1

            result = sp.optimize.minimize(neg_log_likelihood, x0=[2.0, 0.1], bounds=[(1.01, 10), (1e-6, 10)])
            alpha_fit, lambda_fit = result.x
            return alpha_fit, lambda_fit, x_min

        # 1. Ensure strictly positive data
        data = np.asarray(distribution_data)
        data = data[data > 0]
        data = np.sort(data)

        # 2. Fit parameters
        alpha_hat, lambda_hat, min_x = fit_stretched_powerlaw(data)

        # 3. Compute CDF for stretched power-law
        x_grid = np.linspace(min_x, data.max(), 2000)
        cdf_vals = np.array([stretched_powerlaw_cdf(xi, alpha_hat, lambda_hat, min_x) for xi in x_grid])
        inv_cdf = sp.interpolate.interp1d(cdf_vals, x_grid, fill_value="extrapolate")

        # 4. Empirical quantiles from inverse CDF
        quantiles = np.linspace(0.01, 0.99, len(data))
        theoretical_q = inv_cdf(quantiles)
        empirical_q = np.quantile(data, quantiles)

        # 5. QQ Plot on Log-Log scale
        QQPlots.qq_plot(empirical_q, theoretical_q, ax, legend_txt, use_log_scale=True)
        if show_y_label:
            ax.set_ylabel("Empirical Quantiles (data)", fontsize=6)
        ax.set_xlabel("Theoretical Quantiles (Stretched Power Law)", fontsize=6)

        return "Log Q–Q Plot (Power Law w. Expon. Cutoff)"

    @staticmethod
    def conditional_plots(x: np.ndarray, y: np.ndarray, ax: plt.Axes, legend_txt: str, model_type: str, show_y_label: bool = False):
        """
        Conditional log plots for lognormal/power-law/stretched power-law models.

        Args:
            x (np.ndarray): Independent variable
            y (np.ndarray): Dependent variable
            ax (plt.Axes): Matplotlib axis to plot on
            legend_txt (str): Legend text
            model_type (str): Type of model (e.g., "log-norm")
            show_y_label (bool): Whether to show the Y-axis label
        """

        # 1. Fit model
        x_fit = np.linspace(min(x), max(x), 10)
        if model_type == "powerlaw":
            model_name = "Power Law"
            y_fit, params = CurveFitModels.power_law(x, y, x_fit)
        elif model_type == "powerlaw-ec":
            model_name = "Power Law w. Expon. Cutoff"
            y_fit, params = CurveFitModels.stretched_power_law(x, y, x_fit)
        elif model_type == "lognorm":
            model_name = "Lognormal"
            y_fit, params = CurveFitModels.lognormal(x, y, x_fit)
        elif model_type == "gamma":
            model_name = "Gamma"
            y_fit, params = CurveFitModels.gamma(x, y, x_fit)
        else:
            return None

        if y_fit is None:
            y_fit = np.zeros_like(y)

        # 2. Plot
        ax.scatter(np.log(x), np.log(y), s=10, alpha=0.5, label=f"{legend_txt}")
        ax.plot(np.log(x_fit), np.log(y_fit), 'r', label="fitted curve")
        # ax.plot(df["logN"], mu_hat + sigma_hat, 'g--', label="+1σ")
        # ax.plot(df["logN"], mu_hat - sigma_hat, 'g--', label="-1σ")
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=6)
        ax.grid(linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_frame_on(True)  # keep only small subplot borders visible
        if show_y_label:
            ax.set_ylabel("log(data)", fontsize=6)
        ax.set_xlabel("log(Node Count)", fontsize=6)

        return f"Conditional Fit ({model_name})"



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
            df = pd.read_csv(csv_path, delimiter=delimiter)

            mat_label = os.path.splitext(a_file)[0]  # The file name (without extension)
            df.insert(0, "Material", mat_label)

            if mat_label not in all_sheets:
                all_sheets[mat_label] = df
    return all_sheets


def sgt_spider_plot(df_sgt: pd.DataFrame, gt_params: list[str], material_labels: dict|None=None, grid_levels: int = 6) -> None | plt.Figure:
    """
    Generates a spider (radar) plot to compare Graph-Theoretic (GT) parameters
    across multiple material samples, typically derived from SEM images.

    This visualization helps identify similarities or differences in structural
    characteristics among materials based on their GT parameter values.

    Args:
        df_sgt (pd.DataFrame): DataFrame containing - 'Material', 'Parameter', and 'value-1', 'value-2', 'value-3', 'value-4' columns
        gt_params (list[str]): List of GT parameters to plot along the spider axes:

            No. of Nodes -- Number of Nodes,

            No. of Edges -- Number of Edges,

            ND -- Network Diameter,

            GD -- Graph Density,

            AD -- Average Degree,

            GE -- Global Efficiency,

            WI -- Wiener Index,

            ASC -- Assortativity Coefficient,

            ACC -- Average Clustering Coefficient,

            BC -- Average Betweenness Centrality,

            EC -- Average Eigenvector Centrality,

            CC-- Average Closeness Centrality,

            Avg. E. Angle -- Average Edge Angle (degrees),

            Med. E. Angle -- Median Edge Angle (degrees)

        material_labels (dict): Mapping of key--name of the material (whose image is under study). Key should be present in df_sgt DataFrame
        grid_levels (int, optional): Number of levels to use for the grid. Defaults to 6.

    Returns:
        None | matplotlib.figure.Figure:
            The generated Matplotlib Figure if successful, or None if inputs are invalid.
    """

    def compute_grid_scale():
        """Compute grid scale for spider plot"""
        # Below zero levels
        neg_levels = np.linspace(min_val, 0, grid_levels // 4 + 1, endpoint=False)
        # Above zero levels
        pos_levels = np.linspace(0, max_val, grid_levels // 2 + 1)
        # Merge them
        scale_levels = np.concatenate([neg_levels, pos_levels])
        scale_levels = np.sort(scale_levels)
        extra_level = scale_levels[-1] + (scale_levels[-1] - scale_levels[-2])
        scale_levels = np.append(scale_levels, extra_level)
        return scale_levels

    def format_scale_value(value):
        """Format numbers with K, M abbreviations"""
        if abs(value) >= 1_000_000:
            return f'{value / 1_000_000:.1f}M'
        elif abs(value) >= 1_000:
            return f'{value / 1_000:.1f}K'
        elif abs(value) >= 100:
            return f'{value:.0f}'
        elif abs(value) >= 10:
            return f'{value:.1f}'
        else:
            return f'{value:.2f}'

    def shift_value(val):
        """Shift value so that the Spider Plot center is at the minimum value."""
        return val - min_val

    def compute_gt_avg():
        """
        Computes the average and Standard Deviation (if possible) of GT values.
        """

        value_cols = []
        for col in all_df_cols:
            if col.startswith("value"):
                value_cols.append(col)

        if len(value_cols) == 1:
            df_sgt["Avg."] = df_sgt[value_cols].to_numpy()
            return

        # Ensure the value columns exist
        if all(col in df_sgt.columns for col in value_cols):
            df_sgt["Avg."] = df_sgt[value_cols].to_numpy().mean(axis=1)
            df_sgt["Std. Dev."] = df_sgt[value_cols].to_numpy().std(axis=1)
        return

    if df_sgt is None or gt_params is None:
        return None

    all_df_cols = df_sgt.columns.tolist()
    if "value" not in all_df_cols and "value-1" not in all_df_cols:
        return None

    # Get material-labels
    if material_labels is None:
        unique_materials = df_sgt["Material"].unique().tolist()
        material_labels = {str(label): str(label) for label in unique_materials}

    # Rename Columns: apply replacements in the "parameter" column
    param_rename_map = {
        "Number of nodes": "No. of Nodes",
        "Number of edges": "No. of Edges",
        "Network diameter": "ND",
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
    if "parameter" in df_sgt.columns:
        df_sgt["parameter"] = df_sgt["parameter"].replace(param_rename_map)

    # Compute Averages and Standard Deviations
    compute_gt_avg()

    # Filter and pivot
    df_avg = df_sgt.pivot(index='Material', columns='parameter', values='Avg.')
    df_avg = df_avg[gt_params]  # Ensure selected parameters and consistent order
    if 'Std. Dev.' in df_sgt.columns:
        df_std = df_sgt.pivot(index='Material', columns='parameter', values='Std. Dev.')
        df_std = df_std[gt_params]
    else:
        df_std = None

    # Spider plot setup
    num_vars = len(gt_params)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    # Create the figure and axes (regular cartesian, not polar)
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    # Determine max_val based on data
    all_values = []
    for key in material_labels.keys():
        vals = df_avg.loc[key]
        values = np.atleast_1d(vals) # Convert to numpy arrays (handles both scalars and sequences)

        if df_std is not None:
            err = df_std.loc[key]
            errors = np.atleast_1d(err)
            all_values.extend(np.array(values) + np.array(errors))
        else:
            all_values.extend(np.array(values))
    max_val = max(all_values)
    min_val: int = min(-max_val//2, min(all_values))
    levels = compute_grid_scale()

    # Draw grid lines (polygon grid)
    for level in levels:
        shifted_level = shift_value(level)
        x_grid = shifted_level * np.cos(np.append(angles, angles[0]))
        y_grid = shifted_level * np.sin(np.append(angles, angles[0]))
        ax.plot(x_grid, y_grid, 'k-', linewidth=0.5, alpha=0.3)
        # Add the scale value label below the gridline (at the bottom-most point)
        scale_val = format_scale_value(level)
        ax.text(0.1, shifted_level, scale_val, ha='left', va='bottom', fontsize=8, alpha=0.75)

    # Draw axes from the center to each vertex
    max_shifted = shift_value(max(levels))
    for angle in angles:
        ax.plot([0, max_shifted * np.cos(angle)], [0, max_shifted * np.sin(angle)], 'k-', linewidth=0.5, alpha=0.5)

    # Plot each material
    for key, material_name in material_labels.items():
        vals = df_avg.loc[key]
        values = np.atleast_1d(vals)
        values = shift_value(values)

        # Convert to cartesian coordinates
        x = values * np.cos(angles)
        y = values * np.sin(angles)

        # Perpendicular direction vectors for each angle (rotated +90°)
        dx_perp = -np.sin(angles)
        dy_perp = np.cos(angles)

        # Draw the polygon line and capture color
        poly_line, = ax.plot([*x, x[0]], [*y, y[0]], linewidth=1.5)
        color = poly_line.get_color()

        # Fill polygon
        ax.fill(x, y, alpha=0.1, color=color)

        # Draw perpendicular error bars
        if df_std is not None:
            err_vals = df_std.loc[key]
            errors = np.atleast_1d(err_vals)
            err_scale = 5  # increase to make bars longer
            for i in range(len(angles)):
                xi, yi = x[i], y[i]
                err = errors[i] * err_scale

                # Endpoints of perpendicular error bar
                x1 = xi + err * dx_perp[i]
                y1 = yi + err * dy_perp[i]
                x2 = xi - err * dx_perp[i]
                y2 = yi - err * dy_perp[i]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.8)

        # Add label to legend (without the duplicate polygon)
        ax.plot([], [], color=color, label=material_name)

    # Add labels at the vertices
    label_distance = max_shifted * 1.15
    for i, (angle, param) in enumerate(zip(angles, gt_params)):
        x_label = label_distance * np.cos(angle)
        y_label = label_distance * np.sin(angle)

        # Convert angle to degrees for matplotlib rotation
        angle_deg = np.rad2deg(angle)
        check_angle = np.round(angle_deg, 0) % 360

        # Tilt Logic:
        # Flip text if it's on the left side (between 90 and 270 degrees)
        # to keep it right-side up.
        if check_angle == 0 or check_angle == 180:
            display_angle = 90
        elif check_angle == 120 or check_angle == 300:
            display_angle = 30
        elif check_angle == 240 or check_angle == 60:
            display_angle = -30
        else:
            display_angle = 0
        ax.text(
            x_label, y_label, param,
            ha='center', va='center',
            fontsize=14,
            fontweight='bold',
            rotation=display_angle,  # Apply the tilt
            rotation_mode='anchor'  # Ensures rotation is around the text center
        )

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    limit = max_shifted * 1.3
    limit = shift_value(limit)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # Remove axes
    ax.axis('off')

    # Set title and legend position
    if df_std is not None:
        ax.set_title("StructuralGT Spider Plot with STD Error Bars", fontsize=16, pad=20)
    else:
        ax.set_title("StructuralGT Spider Plot", fontsize=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=14)
    fig.tight_layout()
    return fig


def sgt_scaling_plot(y_title: str, df_data: pd.DataFrame, mat_labels: dict, skip_test: bool = False, fit_func: str | None = None, qq_plot: bool = True) -> None | plt.Figure:
    """
    Generates a scaling plot showing error bars for a sample material and displays
    corresponding Kolmogorov–Smirnov test results for different statistical fits (PowerLaw, Exponential, Lognormal).
    The right subplot contains only formatted text (no axes, no borders).

    Args:
        y_title (str): Y-axis title
        df_data (pd.DataFrame): DataFrame containing 'Material', 'x-avg', 'y-avg', 'x-std', and 'y-std'
        mat_labels (dict): Mapping of key--name of the material (whose image is under study). Key should be present in df_data DataFrame
        skip_test (bool, optional): Whether to skip the KS test. Defaults to False
        fit_func (str, optional): Function to fit the data (log-normal, power-law, exponential). Defaults to None
        qq_plot (bool, optional): Whether to plot the QQ plot. Defaults to True

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

        # Define distributions to test
        # stretched_powerlaw = StretchedPowerlawGen(a=0.01, name='stretched_powerlaw')
        distributions = {
            "Power Law": stats.powerlaw,
            "Exponential": stats.expon,
            "Log Normal": stats.lognorm,
            "Gamma": stats.gamma,
            "Weibull": stats.weibull_min,
            #"Stretched Power Law": stretched_powerlaw, ## HAS ERROR
        }

        # Prepare arguments for each parallel process
        data = df_distribution['y-avg'].to_numpy()
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

    def gen_qq_plot(model_name: str|None) -> str:
        """
        Generate Q–Q or conditional plots for different models.

        Args:
            model_name (str|None): One of {"lognorm", "powerlaw", "powerlaw-ec"}.

        Returns:
            str: Axis title (model name or QQ plot type).
        """

        if fit_func is None or model_name is None:
            return ""

        # Map each model to its respective QQ function
        qq_funcs = {
            "lognorm": QQPlots.log_qq_plot,
            "gamma": QQPlots.gamma_qq_plot,
            "powerlaw": QQPlots.pwr_qq_plot,
            "powerlaw-ec": QQPlots.stretched_pwr_qq_plot
        }

        # Check model name validity
        if model_name not in qq_funcs:
            return ""

        # Choose axis
        ax = ax_4_grids[i]

        # Run QQ plot or conditional fit
        if qq_plot:
            ax_title = qq_funcs[model_name](y_avg, ax, material_name, is_left)
        else:
            ax_title = QQPlots.conditional_plots(x_avg, y_avg, ax, material_name, model_name, is_left)
            ax_title = "" if ax_title is None else ax_title

        return ax_title

    if y_title is None or df_data is None or mat_labels is None:
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
    ax_4_grids, i, ax_4_title = [], 0, ""
    if type(fit_func) == str:
        ax_3 = fig.add_subplot(gs[1, 0])     # Curve fits with selected distributions

        if fit_func in ["lognorm", "powerlaw", "powerlaw-ec", "gamma"]:
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
    for key, material_name in mat_labels.items():
        df_sample = df_data[df_data['Material'] == key].copy()
        if df_sample.empty:
            continue

        # Perform the Goodness-of-fit test?
        if not skip_test:
            # KS tests for different fits
            txt_test += parallel_goodness_of_fit(df_sample, material_name)

        # Plot Curves fitted to specific distributions
        if ax_3 is not None:
            x_avg = pd.Series(df_sample['x-avg']).to_numpy()
            y_avg = pd.Series(df_sample['y-avg']).to_numpy()
            x_fit = np.linspace(min(x_avg), max(max(x_avg), 10000), 100)
            y_fit, axis_label = None, ""
            if fit_func == "lognorm":
                y_fit, params = CurveFitModels.lognormal(x_avg, y_avg, x_fit)
                mu_fit, sigma_fit, a_log_fit = params["mu"], params["sigma"], params["a"]
                axis_label = f'{material_name}: a={a_log_fit:.2f}, $\\mu={mu_fit:.3f}$, $\\sigma={sigma_fit:.3f}$'
            elif fit_func == "gamma":
                y_fit, params = CurveFitModels.gamma(x_avg, y_avg, x_fit)
                a_fit, alpha_fit, theta_fit = params["a"], params["alpha"], params["theta"]
                axis_label = f'{material_name}: a={a_fit:.2f}, $\\alpha={alpha_fit:.2f}, \\theta={theta_fit:.2f}$'
            elif fit_func == "powerlaw":
                y_fit, params = CurveFitModels.power_law(x_avg, y_avg, x_fit)
                a_fit, k_fit = params["a"], params["k"]
                axis_label = f'{material_name}: $a={a_fit:.3f}, k={k_fit:.3f}$'
            elif fit_func == "powerlaw-ec":
                y_fit, params = CurveFitModels.stretched_power_law(x_avg, y_avg, x_fit)
                a_fit, k_fit, cut_fit, beta_fit = params["a"], params["k"], params["x_c"], params["beta"]
                axis_label = f'{material_name}: $a={a_fit:.3f}, k={k_fit:.3f}, x_c={cut_fit:.3f}, \\beta={beta_fit:.3f}$'
            elif fit_func == "linear":
                y_fit, params = CurveFitModels.linear(x_avg, y_avg, x_fit)
                slope_fit, intercept_fit = params["m"], params["b"]
                axis_label = f'{material_name}: $slope={slope_fit:.3f}, b={intercept_fit:.3f}$'
            elif fit_func == "exponential":
                y_fit, params = CurveFitModels.exponential(x_avg, y_avg, x_fit)
                a_fit, b_fit, c_fit = params["a"], params["lambda"], params["c"]
                axis_label = f'{material_name}: $a={a_fit:.3f}, b={b_fit:.3f}$'
            ax_3.plot(x_fit, y_fit, label=axis_label, linestyle='-') if y_fit is not None else None

            if i < len(ax_4_grids):
                is_left = True if i in (0, 2) else False
                ax_4_title = gen_qq_plot(fit_func)
                i += 1

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
    ax_1.set_xlabel('Node Count', fontsize=12)
    ax_1.set_ylabel(y_title, fontsize=12)
    ax_1.set_title(f'Nodes vs {y_title} (Actual Data)', fontsize=14)
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
        elif fit_func == "gamma":
            ax_3.set_title(
                r"Gamma Fit: $y = a \cdot \frac{x^{\alpha-1}}{\Gamma(\alpha)}\cdot \frac{\exp(-\frac{x}{\theta})}{"
                r"\Gamma(\alpha)\theta^{\alpha}}$"
                f"\nNodes vs {y_title}",
                fontsize=10
            )
        elif fit_func == "powerlaw":
            ax_3.set_title(
                r"PowerLaw Fit: $y = a \cdot x^{-k}$"
                f"\nNodes vs {y_title}",
                fontsize=10
            )
        elif fit_func == "powerlaw-ec":
            ax_3.set_title(
                r"Stretched PowerLaw Fit: $y = a \cdot x^{-k} \cdot \exp((\frac{-x}{x_c})^{\beta})$"
                f"\nNodes vs {y_title}",
                fontsize=10
            )
        elif fit_func == "linear":
            ax_3.set_title(
                r"Linear Fit: $y = m(x) + b$"
                f"\nNodes vs {y_title}",
                fontsize=10
            )
        elif fit_func == "exponential":
            ax_3.set_title(
                r"Exponential Fit: $y = a \cdot e^{-\frac{x}{\lambda}} + c$"
                f"\nNodes vs {y_title}",
                fontsize=10
            )
        else:
            fig.tight_layout()
            return fig
        ax_3.set_xlabel('Node Count', fontsize=12)
        ax_3.set_ylabel(y_title, fontsize=12)
        ax_3.legend(frameon=False, fontsize=8)
        ax_3.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)  # cleaner grid

        if ax_4 is not None:
            ax_4.axis("off")  # hide all ticks and labels
            ax_4.set_frame_on(False)  # remove the border/frame
            ax_4.set_title(f"{ax_4_title}: {y_title}")

    fig.tight_layout()
    return fig
