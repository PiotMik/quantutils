import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import plotly.graph_objects as go
from typing import Callable, Optional
plt.style.use('ggplot')


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
    MC = 10000
    n_steps = 10000
    dt = 1/n_steps
    simulations = [euler_maruyama(x0=0.05,
                                  a_t=lambda t, x: mu,
                                  b_t=lambda t, x: sigma,
                                  n_steps=n_steps,
                                  dt=dt)[1] for i in range(MC)]

    d = pd.DataFrame(simulations).transpose()

    # Distribution plot t = 1
    d.plot(legend=False, color='teal', alpha=0.1)
    plt.title("IR in the Merton model - simulated path")

    finite_distribution = np.array(d.iloc[-1, :])
    ax = sns.histplot(finite_distribution, stat='density', kde=True, color='teal')

    x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
    x_pdf = np.linspace(x0, x1, 100)
    y_pdf = scipy.stats.norm.pdf(x_pdf, loc=mu, scale=sigma)

    ax.plot(x_pdf, y_pdf, color='orange', lw=2, label=f'N({mu}, {sigma})')
    plt.title("")
    ax.legend()

    #
    t = np.linspace(0, n_steps*dt, n_steps)
    x = np.linspace(d.min().min(), d.max().max(), 100)
    dens = np.zeros((n_steps, 100))
    for index, row in d.iterrows():

        kde = scipy.stats.gaussian_kde(row)
        density = kde.pdf(x)
        dens[index, :] = density
    fig = go.Figure(data=[go.Surface(z=dens[300::100, :], x=x, y=t[300::100])])
    fig.update_layout(title='Density of simulated paths', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()