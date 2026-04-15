from matplotlib import pyplot as plt
from .likelihood import *
from .solvers import *

def plot_density_from_points(X, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.hist2d(X.cpu().detach().numpy()[:,0], X.cpu().detach().numpy()[:,1], bins=150)
    ax.set_aspect('equal')

