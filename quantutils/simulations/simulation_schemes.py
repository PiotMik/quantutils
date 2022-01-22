import numpy as np
from typing import Callable, Optional


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
    pass