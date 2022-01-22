import numpy as np
from ..simulations.simulation_schemes import euler_maruyama
from typing import Optional


def merton(mu: float, sigma: float, x0: float,
           n_steps: int, dt: float,
           random_toss: Optional[np.array] = None):
    """
    Generate an interest rate trajectory from merton model, using Euler-Maruyama scheme.
    Model:
        dX(t) = mu*dt + sigma*dWt
    Parameters
    ----------
    mu: float,
     drift parameter
    sigma: float,
     diffusion parameter
    x0: float,
     starting value
    n_steps: int,
     number of grid points to simulate on
    dt: float,
     distance between the grid points
    random_toss: Optional[np.array],
     optional array of +1 and -1 to simulate Wiener Process. If not passed, will be generated automatically.
    """

    return euler_maruyama(x0,
                          a_t=lambda t, x: mu,
                          b_t=lambda t, x: sigma,
                          n_steps=n_steps,
                          dt=dt,
                          random_toss=random_toss)


def vasicek(a: float, b: float, sigma: float, x0: float,
            n_steps: int, dt: float,
            random_toss: Optional[np.array] = None):
    """
    Generate an interest rate trajectory from merton model, using Euler-Maruyama scheme.
    Model:
        dX(t) = [a-b*X(t)]*dt + sigma*dWt
    Parameters
    ----------
    a: float,
     asymptotic, long term level (mean-reversion)
    b: float,
     speed of mean reversion
    sigma: float,
     diffusion parameter
    x0: float,
     starting value
    n_steps: int,
     number of grid points to simulate on
    dt: float,
     distance between the grid points
    random_toss: Optional[np.array],
     optional array of +1 and -1 to simulate Wiener Process. If not passed, will be generated automatically.
    """

    return euler_maruyama(x0,
                          a_t=lambda t, x: a - b*x,
                          b_t=lambda t, x: sigma,
                          n_steps=n_steps,
                          dt=dt,
                          random_toss=random_toss)


if __name__ == '__main__':
    pass
