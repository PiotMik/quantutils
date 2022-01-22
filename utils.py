from stochastic_models import gbm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

if __name__ == '__main__':

    np.random.seed(123)
    gbm_paths = gbm(100, 0.01, 0.3, 1, 100, 3)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()
    plt.xlim(0, gbm_paths.shape[0])
    plt.ylim(gbm_paths.min(), gbm_paths.max())
    plt.title(f"GBM, {gbm_paths.shape[1]} path(s)")


    def animate_trajectory(i):
        data = pd.DataFrame(gbm_paths[:int(i + 1), :])
        p = sns.lineplot(x=data.index, y = data.iloc[:, 1], color = 'r')
        plt.setp(p.lines)

    def augment(xold, yold, numsteps):
        xnew=[]
        ynew=[]
        for i in range(len(xold) - 1):
            difX = xold[i+1]- xold[i]
            stepsX= difX/numsteps
            difY = yold[i+1]- yold[i]
            stepsY= difY/numsteps
            for s in range(numsteps):
                xnew = np.append(xnew, xold[i] + s*stepsX)
                ynew = np.append(ynew, yold[i] + s*stepsY)
        return xnew, ynew

    ani = animation.FuncAnimation(fig, animate_trajectory, frames=gbm_paths.shape[0], repeat = True)
    ani.save('Manim.mp4', writer = writer)