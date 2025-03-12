from shapely.geometry import MultiPoint
import flopy
import numpy as np
import sys
sys.path.append('..')
from dependencies.plotting.movie import movie
from dependencies.convert_transient import convert_to_transient

def transient_run(pars):
    #%% load parameters, directories, etc.
    
    mname       = pars['mname']
    model_dir   = pars['trs_ws']
    obsxy       = pars['obsxy'] 
    nsteps      = int(pars['nsteps'])
    
    sim         = convert_to_transient(model_dir, pars, nsteps)
    gwf         = sim.get_model(mname)
    
    #%% run simulation
    sim.run_simulation()
    
    #%% Generate Observations
    
    heads       = np.empty((nsteps, 1, gwf.disv.ncpl.data))
    for i in range(nsteps):
        heads[i,0,:] = gwf.output.head().get_data(kstpkper=(0, i))
        
    ixs         = flopy.utils.GridIntersect(gwf.modelgrid, method = "vertex")
    result      = ixs.intersect(MultiPoint(obsxy))
    
    obs         = {}
    for i, cellid in enumerate(result.cellids):
        obs[i] = {'cellid':cellid,
                  'h_obs':np.empty((1,nsteps))}
        
    for i in range(nsteps):
        for j in range(len(obs)):
            obs[j]['h_obs'][0,i] = heads[i,0,obs[j]['cellid']]
    

    np.save(pars['vr_h_d'], heads)
    np.save(pars['vr_o_d'], obs, allow_pickle=True)
    
    # movie(gwf, diff = False, contour = True)
 