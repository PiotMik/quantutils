import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from quantutils.simulations.simulation_schemes import milstein, euler_maruyama

if __name__ == '__main__':

    x0 = 100.
    mu = .05
    sigma = .2
    a_t = lambda t, x: x * mu
    b_t = lambda t, x: x * sigma
    bx_t = lambda t, x: sigma

    n_steps = 100
    n_paths = 1000
    T = 1.0

    t, St = milstein(x0=x0, n_paths=n_paths, n_steps=n_steps, dt=T / n_steps,
                     a_t=lambda t, x: x * mu,
                     b_t=lambda t, x: x * sigma,
                     bx_t=lambda t, x: sigma)

    _, ax = plt.subplots(1, 2)
    ax[0].plot(t, St)
    sns.distplot(St[-1, :], ax=ax[1], kde=True, vertical=True)
    plt.suptitle('Derivative provided')
    plt.tight_layout()
    plt.show()

    t, St = milstein(x0=x0, n_paths=n_paths, n_steps=n_steps, dt=T / n_steps,
                     a_t=lambda t, x: x * mu,
                     b_t=lambda t, x: x * sigma)

    _, ax = plt.subplots(1, 2)
    ax[0].plot(t, St)
    sns.distplot(St[-1, :], ax=ax[1], kde=True, vertical=True)
    plt.suptitle('Derivative free')
    plt.tight_layout()
    plt.show()
