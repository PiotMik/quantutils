import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Optional
plt.style.set('ggplot')


def euler_maruyama(x0: float, a_t: Callable, b_t: Callable, n_steps: int, dt: float, random_toss: Optional[np.array] = None):

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


if __name__ == "__main__":

    mu = 0.01
    sigma = 0.6

    t, xt = euler_maruyama(x0=0.05,
                           a_t=lambda t, x: mu,
                           b_t=lambda t, x: sigma,
                           n_steps=10000,
                           dt=0.0001)
    MC = 100
    simulations = [euler_maruyama(x0=0.05,
                                  a_t=lambda t, x: mu,
                                  b_t=lambda t, x: sigma,
                                  n_steps=10000,
                                  dt=0.0001)[1] for i in range(MC)]
    d = pd.DataFrame(simulations).transpose()
    d.plot(legend=False, color='teal', alpha=0.1)
    plt.title("IR in the Merton model - simulated path")

    finite_distribution = d.iloc[-1, :]
