# -*- coding: utf-8 -*-
"""
This script visualizes the hydraulic conductivity field and the recharge field
of a MODFLOW 6 model

@author: Janek Geiger
"""

import flopy
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_fields(gwf: flopy.mf6.modflow.mfgwf.ModflowGwf, pars,  logk_proposal, rech_proposal: np.ndarray, POI = []):
    
    kmin    = np.min(logk_proposal)
    kmax    = np.max(logk_proposal[logk_proposal < -0.5])
    rmin    = np.min(np.loadtxt(pars['r_r_d'], delimiter = ','))
    rmax    = np.max(np.loadtxt(pars['r_r_d'], delimiter = ','))
    
    pad = 0.1
    
    rch_spd     = gwf.rch.stress_period_data.get_data()
    rch_spd[0]['recharge'] = rech_proposal
    gwf.rch.stress_period_data.set_data(rch_spd)
  
    # Define font size
    fontsize = 20
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))  # Increase figsize height and width
    
    # Upper plot
    ax0 = flopy.plot.PlotMapView(model=gwf, ax=axes[0])
    c0 = ax0.plot_array(logk_proposal, cmap=cm.bilbao_r, alpha=1, vmin = kmin, vmax = kmax)
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=pad)  # Adjust size and pad for better spacing
    cbar0 = fig.colorbar(c0, cax=cax0)
    cbar0.mappable.set_clim(kmin, kmax)
    cbar0.set_label('Log-Conductivity (log(m/s))', fontsize=fontsize)
    cbar0.ax.tick_params(labelsize=fontsize)
    axes[0].set_aspect('equal')  # Change to 'auto' to prevent squishing
    axes[0].set_ylabel('Y-axis', fontsize=fontsize)
    axes[0].tick_params(axis='both', which='major', labelsize=fontsize)
    if len(POI) > 0:
        axes[0].scatter(POI[:,0], POI[:,1])
        axes[0].scatter(pars['welxy'][:,0], pars['welxy'][:,1], c = 'black')
    
    # Lower plot
    ax1 = flopy.plot.PlotMapView(model=gwf, ax=axes[1])
    c1 = ax1.plot_array(rech_proposal, cmap=cm.turku_r, alpha=1)
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=pad)  # Adjust size and pad for better spacing
    cbar1 = fig.colorbar(c1, cax=cax1)
    cbar1.mappable.set_clim(rmin, rmax)
    cbar1.set_label('Recharge (m/s)', fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)
    axes[1].set_aspect('equal')  # Change to 'auto' to prevent squishing
    axes[1].set_ylabel('Y-axis', fontsize=fontsize)
    axes[1].set_xlabel('X-axis', fontsize=fontsize)
    axes[1].tick_params(axis='both', which='major', labelsize=fontsize)
    
    plt.tight_layout(pad=pad)  # Adjust the overall padding between subplots
    plt.show()