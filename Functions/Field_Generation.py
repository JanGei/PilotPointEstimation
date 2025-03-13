# -*- coding: utf-8 -*-
"""
This script creates the reference hydraulic conductivity field and the 
reference recharge field for the virtual truth.

@author: Janek Geiger
"""
import numpy as np
import os
import flopy
from Functions.randomK import randomK
from scipy.interpolate import griddata
# from model_params import get

def generate_fields(pars):

# if __name__ == '__main__':
#     #%% Loading parameters and model
#     pars = get()
    lx      = pars['lx']
    ang     = pars['ang']
    sigma   = pars['sigma']
    sim_ws  = pars['sim_ws']
    mname   = pars['mname']
    
    sim        = flopy.mf6.modflow.MFSimulation.load(
                            version             = 'mf6', 
                            exe_name            = 'mf6',
                            sim_ws              = sim_ws, 
                            verbosity_level     = 0
                            )
    
    gwf = sim.get_model(mname)
    mg  = gwf.modelgrid
    cxy = np.vstack((mg.xyzcellcenters[0], mg.xyzcellcenters[1])).T

    #%% Field generation
    x = np.arange(pars['dx'][0]/2, pars['nx'][0]*pars['dx'][0], pars['dx'][0])
    y = np.arange(pars['dx'][1]/2, pars['nx'][1]*pars['dx'][1], pars['dx'][1])

    # Grid in Physical Coordinates
    X, Y = np.meshgrid(x, y)
    
    K = randomK(np.deg2rad(ang[0]), sigma[0], pars, ftype = 'K', random = False)
    
    # Choosing between isotropic and anisotropic recharge
    if pars['rch_is']:
        Rflat = np.array([pars['mu'][1]])
    else:
        R = randomK(np.deg2rad(ang[1]), sigma[1], pars, ftype = 'R', random = False)
        Rflat =  griddata((X.ravel(order = 'F'), Y.ravel(order = 'F')), R.ravel(order = 'F'),
                         (cxy[:,0], cxy[:,1]), method='nearest')
    
    Kflat =  griddata((X.ravel(order = 'F'), Y.ravel(order = 'F')), K.ravel(order = 'F'),
                     (cxy[:,0], cxy[:,1]), method='nearest')
    
    #%% Saving the fields - in m/s
    np.savetxt(os.path.join(pars['k_r_d']), Kflat, delimiter = ',')
    np.savetxt(os.path.join(pars['r_r_d']), Rflat/1000/86400, delimiter = ',')
