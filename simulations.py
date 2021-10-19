import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd


def gbm(s0: float, r: float, sigma: float, t: float, time_steps: int, n_paths: int):
    """
    Generate Geometric Brownian Motion paths
    :param s0: float, starting price
    :param r: float, risk free rate
    :param sigma: float, volatility
    :param t: float, time horizon
    :param time_steps: int, how many subintervals include from 0 to T
    :param n_paths: int, how many paths to simulate
    :return: paths: ndarray, simulated paths
    """
    dt = float(t) / time_steps
    paths = np.zeros((time_steps + 1, n_paths), np.float64)
    paths[0] = s0
    for t in range(1, time_steps + 1):
        rand = np.random.standard_normal(n_paths)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)
    return paths


if __name__ == '__main__':

    gbm_paths = gbm(100, 0.05, 0.2, 5, 1000, 1000)
    t = np.linspace(0, 5, 1000 + 1)
    fig, ax = plt.subplots()
    ax.plot(t, gbm_paths, color='blue', alpha = 0.1, label = None)
    expected_value = gbm_paths[-1, :].mean()
    ax.scatter(t[-1], expected_value, color='red', label = 'E[S(T)]')
    ax.legend()
