"""
Functions for common plots across different types of simulations.
"""

import math
import matplotlib.lines as lines
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


def plot_density(arr, axis, title=None, boundaries=None):
    
    if title is not None:
        axis.set_title(title, wrap=True)
        
    if boundaries is not None:
        for b in boundaries:
            if isinstance(b, lines.Line2D):
                axis.add_line(b)
            elif isinstance(b, Rectangle):
                axis.add_patch(b)

    axis.imshow(arr.T)
    axis.invert_yaxis()
    
def plot_decay(arr, label_x, label_y, title=None, init_arc=None, path=None, nt_log=1):
    if init_arc is not None:
        arr = np.insert(arr, 0, init_arc, axis=0)
        
    slen = arr.shape[0]
    subplot_columns = 4
    subplot_rows = math.ceil(slen/subplot_columns)
    fig, axes = plt.subplots(subplot_rows, subplot_columns)
    
    plt.gcf().set_size_inches(subplot_columns*4,subplot_rows*3)
    plt.setp(axes, xlabel=label_x, 
             ylabel=label_y, 
             ylim=(np.min(arr), np.max(arr)))
    if title:
        plt.suptitle(title, wrap=True)
    for i in range(slen):
        axis = axes[i//subplot_columns, i%subplot_columns]
        if init_arc is not None and i==0:
            axis.plot(arr[i], color='red')
            axis.set_title("Initial value plot")
        else:
            axis.plot(arr[i])
            if init_arc is not None:
                axis.set_title(f"Time Step {(i-1)*nt_log}")
            else:
                axis.set_title(f"Time Step {i*nt_log}")
    fig.tight_layout(h_pad=2,w_pad=2)
    plt.show(block=False)
    if path:
        plt.savefig(f"{path}/Decay-{label_x}-{label_y}.png")
        plt.clf()
        plt.cla()
        plt.close()
            
def plot_amplitude(arr, label_x, label_y, title=None, path=None):
    plt.plot(arr)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if title:
        plt.title(title, size=10, wrap=True)
    plt.tight_layout()
    plt.show(block=False)
    if path:
        plt.savefig(f"{path}/Amplitude-{label_x}-{label_y}.png")
        plt.clf()
        plt.cla()
        plt.close()

def plot_combined(periodic_data,
                  label_x, label_y, path,
                  reference=None, 
                  title=None):
    if reference is not None:
        plt.plot(reference, color='black', label="Analytical")
        plt.legend()

    for i in range(periodic_data.shape[0]):
        plt.plot(periodic_data[i])
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if title:
        plt.title(title, size=10, wrap=True)
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(f"{path}/Combined_plot-{label_x}-{label_y}.png")
    plt.clf()
    plt.cla()
    plt.close()
