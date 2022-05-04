import numpy as np
from typing import Callable, Optional


def euler_maruyama(x0: float, a_t: Callable, b_t: Callable, n_steps: int, dt: float, random_toss: Optional[np.array] = None):
        """
        Simulate a stochastic process according to Euler-Maruyama method.

        Parameters
        ----------
        x0: float
            Starting value for the process
        a_t: Callable f: (t, x) -> R
            First characteristic of a stochastic process,
        b_t: Callable f: (t, x) -> R
            Second characteristic of a stochastic process
        n_steps: int
            Number of time discretization steps
        dt: float
            Lenght of a time step
        random_toss: Optional[np.array]
            Array of +1/-1, which is used for Wiener Process approximation.
            If not provided, it gets generated automatically.

        Returns
        -------
        (np.array, np.array)
            Time array and simulated stochastic process values
        """
        if not random_toss:
            random_toss = np.random.rand(n_steps).round()*2 - 1
        if random_toss.shape != (n_steps, ):
            raise ValueError("Incompatible shapes")

        xt = [x0]
        t = [0.]
        for k in range(1, n_steps):
            t_k = k*dt
            x_k = xt[-1] + a_t(t[-1], xt[-1])*dt + b_t(t[-1], xt[-1])*random_toss[k-1]*np.sqrt(dt)
            xt.append(x_k)
            t.append(t_k)
        return np.array(t), np.array(xt)

def milstein(x0: float, a_t: Callable, b_t: Callable,
             n_steps: int, n_paths: int, dt: float,
             bx_t: Optional[Callable] = None,
             rnorm: Optional = None):

    if bx_t is None:
        bx_t = lambda t, x: (b_t(t, x + dt) - b_t(t, x))/dt

    if rnorm is None:
        rnorm = np.random.normal(size=(n_steps - 1, n_paths))

    t = np.zeros(shape=(n_steps,))
    St = np.zeros(shape=(n_steps, n_paths))
    St[0, :] = x0

    for i, Z in enumerate(rnorm):
        St[i + 1,] = St[i,] + a_t(t[i], St[i,]) * dt + b_t(t[i], St[i,]) * Z * np.sqrt(dt) + \
                     bx_t(t[i], St[i,]) * b_t(t[i], St[i,]) * 0.5 * ((Z * np.sqrt(dt)) ** 2 - dt)
        t[i + 1] = t[i] + dt

    return t, St

if __name__ == "__main__":
    pass