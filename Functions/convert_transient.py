import flopy
from Functions.copy import copy_model
import numpy as np

def convert_to_transient(model_dir: str, target_dir: str, pars: dict, nsteps: int = 1):
    '''
    Copies a steady-state Modflow 6 model and converts the copy into a 
    transient model. The original model does not need to be in steady-state.
    This function can also be used to change the number of stress periods in
    a modflow 6 model.

    Parameters
    ----------
    model_dir : str
        Directory of the steady-state model.
    target_dir : str
        Destination directory of the new model.
    pars : dict
        Simulation / model parameters.
    nsteps : int, optional
        Number of stress periods in the model. The default is 1.

    Returns
    -------
    sim : flopy object
        Simulation object of flopy.

    '''
    
    #%% Loading model parameters, copying model, loading new model
    sname               = pars['sname']
    mname               = pars['mname']
    sfac                = np.genfromtxt(pars['sf_d'],delimiter = ',', names=True)['Wert']
    r_ref               = np.loadtxt(pars['r_r_d'], delimiter = ',')
    rivh                = np.genfromtxt(pars['rh_d'],delimiter = ',', names=True)['Wert']
    welq                 = pars['welq'] 
    welst               = pars['welst'] 
    welnd               = pars['welnd'] 
    
    
    copy_model(model_dir, target_dir)
    
    sim = flopy.mf6.MFSimulation.load(sname,
                                      sim_ws = target_dir,
                                      verbosity_level = 1) 
    #%% Set transient forcing
    gwf             = sim.get_model(mname)
    rch             = gwf.rch
    riv             = gwf.riv
    sto             = gwf.sto
    tdis            = sim.tdis
    wel             = gwf.wel

    # get stressperioddata
    rch_spd         = rch.stress_period_data.get_data()
    riv_spd         = riv.stress_period_data.get_data()
    wel_spd         = wel.stress_period_data.get_data()

    sto_tra         = {}
    sto_sts         = {}
    wel_list        = {}
    riv_list        = {}
    rch_list        = {}
    perioddata      = []


    time_d          = 0
    for i in range(int(nsteps)):
        # recharge
        rch_list[i] = rch_spd[0].copy()
        rch_list[i]['recharge'] = abs(np.array(r_ref).flatten()) * sfac[i]
        # river
        riv_list[i] = riv_spd[0].copy()
        riv_list[i]['stage'] += rivh[i]-rivh[0]
        # wel
        wel_list[i] = wel_spd[0].copy()
        for j in range(len(wel_list[i]['q'])):
            if welst[j] <= time_d and welnd[j] > time_d:
                wel_list[i]['q'][j] = -welq[j]
            else:
                wel_list[i]['q'][j] = 0
        # sto & tdis
        sto_tra[i] = True
        sto_sts[i] = False
        perioddata.append([60*60*6, 1, 1.0])
        
        time_d += 0.25

    #%% set new data 
        
    tdis.perioddata.set_data(perioddata)
    tdis.nper.set_data(len(perioddata))
    sto.transient.set_data(sto_tra)
    sto.steady_state.set_data(sto_sts)
    rch.stress_period_data.set_data(rch_list)
    riv.stress_period_data.set_data(riv_list)
    wel.stress_period_data.set_data(wel_list)

    sim.write_simulation()
    
    return sim
    
