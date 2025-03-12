import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def ellipses(cov_data, mean_cov, pars):

    center = (0, 0)  # center coordinates
    l = 4000

    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    for i, data in enumerate(cov_data):

        ellipse = patches.Ellipse(center,
                                  data[0]*2,
                                  data[1]*2,
                                  angle=np.rad2deg(data[2]),
                                  fill=False,
                                  color='black',
                                  alpha = 0.5,
                                  zorder = 1)
        ax.add_patch(ellipse)

    
    ellipse = patches.Ellipse(center,
                              mean_cov[0]*2,
                              mean_cov[1]*2,
                              angle=np.rad2deg(mean_cov[2]),
                              fill=False,
                              color='blue',
                              zorder = 2)
    ax.add_patch(ellipse)
    
    ellipse = patches.Ellipse(center,
                              pars['lx'][0][0]*2,
                              pars['lx'][0][1]*2,
                              angle=pars['ang'][0],
                              fill=False,
                              color='red',
                              zorder = 2)
    ax.add_patch(ellipse)

    # Set axis limits
    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)
    
    # Set equal aspect ratio for the axis
    ax.set_aspect('equal')
    
    # Display the plot
    plt.grid(True)
    plt.show()