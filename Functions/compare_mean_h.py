import matplotlib.pyplot as plt
import flopy
from cmcrameri import cm

def compare_mean_true_head(gwf, h_fields, poi):
    
    hmin = 12
    hmax = 29
    
    varmin = 0
    varmax = 1e-3
    
    ratmin = -0.25
    ratmax = 0.25
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize= (16,6))

    # Plot upper left
    ax0 = axes[0,0]
    axf0 = flopy.plot.PlotMapView(model=gwf, ax=ax0)
    c0 = axf0.plot_array(h_fields[0], cmap=cm.oslo, alpha=1)
    ax0.set_aspect('equal')


    # Plot upper right 
    ax1 = axes[1,0]
    axf1 = flopy.plot.PlotMapView(model=gwf, ax=ax1)
    axf1.plot_array(h_fields[1], cmap=cm.oslo, alpha=1, vmin=hmin, vmax=hmax)
    ax1.set_aspect('equal')


    # Lower Left
    ax2 = axes[0,1]
    axf2 = flopy.plot.PlotMapView(model=gwf, ax=ax2)
    c2 = axf2.plot_array(h_fields[2], cmap=cm.roma, alpha=1, vmin=varmin, vmax=varmax)
    ax2.set_aspect('equal')
    
    # Lower Right
    ax3 = axes[1,1]
    axf3 = flopy.plot.PlotMapView(model=gwf, ax=ax3)
    c3 = axf3.plot_array(h_fields[1]-h_fields[0], cmap=cm.roma, alpha=1, vmin=ratmin, vmax=ratmax)
    ax3.set_aspect('equal')


    # Add colorbars
    cbar0 = fig.colorbar(c0, ax=[ax0, ax1], fraction=0.1, pad=0.05)
    cbar0.set_label('Head [m]')
    cbar1 = fig.colorbar(c2, ax=ax2, fraction=0.1, pad=0.05, aspect=10)
    cbar1.set_label('Variance')
    cbar2 = fig.colorbar(c3, ax=ax3, fraction=0.1, pad=0.05, aspect=10)
    cbar2.set_label('Ratio')

    
    # Set custom bounds for colorbars
    cbar0.mappable.set_clim(vmin=hmin, vmax=hmax)
    cbar1.mappable.set_clim(vmin=varmin, vmax=varmax)
    cbar2.mappable.set_clim(vmin=ratmin, vmax=ratmax)
    
    if poi.any():
        ax0.scatter(poi[:,0], poi[:,1], c = 'black', marker = 'x', s = 3)
        ax1.scatter(poi[:,0], poi[:,1], c = 'black', marker = 'x', s = 6)
        ax2.scatter(poi[:,0], poi[:,1], c = 'black', marker = 'x', s = 9)
        
    plt.show()    