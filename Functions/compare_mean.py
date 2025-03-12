import matplotlib.pyplot as plt
import numpy as np
import flopy
from cmcrameri import cm

def compare_mean_true(gwf, k_fields, poi):
    
    kmin = np.min(np.log10(k_fields[0]))
    kmax = np.max(np.log10(k_fields[0]))
    vmin = 0
    vmax = 2
    
    
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex='col', sharey=True, figsize= (16,6))

    # Plot upper left - Here come the true - non-log field
    ax0 = axes[0]
    axf0 = flopy.plot.PlotMapView(model=gwf, ax=ax0)
    c0 = axf0.plot_array(np.log10(k_fields[0]), cmap=cm.bilbao_r, alpha=1,vmin=kmin, vmax=kmax)
    ax0.set_aspect('equal')


    # Plot upper right - here comes the meanlogfield
    ax1 = axes[1]
    axf1 = flopy.plot.PlotMapView(model=gwf, ax=ax1)
    axf1.plot_array(k_fields[1]/ np.log(10), cmap=cm.bilbao_r, alpha=1,vmin=kmin, vmax=kmax)
    ax1.set_aspect('equal')


    # Plot lower
    ax2 = axes[2]
    axf2 = flopy.plot.PlotMapView(model=gwf, ax=ax2)
    c2 = axf2.plot_array(k_fields[2], cmap=cm.roma, alpha=1,vmin=vmin, vmax=vmax)
    ax2.set_aspect('equal')
    
    # ax2 = axes[2]
    # gwf.npf.k.set_data((k_fields[1])/ np.log((k_fields[0])))
    # axf2 = flopy.plot.PlotMapView(model=gwf, ax=ax2)
    # c2 = axf2.plot_array((gwf.npf.k.array), cmap=cm.roma, alpha=1)
    # ax2.set_aspect('equal')


    # Add colorbars
    cbar0 = fig.colorbar(c0, ax=[ax0, ax1], fraction=0.1, pad=0.01)
    cbar0.set_label('Log(K)')
    cbar1 = fig.colorbar(c2, ax=ax2, fraction=0.1, pad=0.01, aspect=10)
    cbar1.set_label('Ratio')

    
    # Set custom bounds for colorbars
    cbar0.mappable.set_clim(vmin=kmin, vmax=kmax)
    # cbar1.mappable.set_clim(vmin=0, vmax=1.5)
    # cbar1.mappable.set_clim(vmin=0.5, vmax=1.5)
    
    if poi.any():
        ax0.scatter(poi[:,0], poi[:,1], c = 'black', marker = 'x', s = 3)
        ax1.scatter(poi[:,0], poi[:,1], c = 'black', marker = 'x', s = 6)
        ax2.scatter(poi[:,0], poi[:,1], c = 'black', marker = 'x', s = 9)
        
    plt.show()    