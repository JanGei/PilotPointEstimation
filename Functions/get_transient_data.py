import numpy as np

def get_transient_data(pars: dict, t_step: int):
    
    t_step = int(t_step%(365*24/6))
    r_ref = np.loadtxt(pars['r_r_d'], delimiter = ',')
    sfac  = np.genfromtxt(pars['sf_d'],delimiter = ',', names=True)['Wert']
    welst = pars['welst'] 
    welnd = pars['welnd'] 
    welq  = pars['welq'] 
    rivh  = np.genfromtxt(pars['rh_d'],delimiter = ',', names=True)['Wert']

    rch_data = abs(np.array(r_ref).flatten()) * sfac[t_step]
    riv_delta = rivh[t_step]-rivh[0]
    
    # check if we need to set up new wel files
    day = 0.25 * t_step
    
    if int(day) in pars['welst'] or int(day) in pars['welnd']:
        
        wel_data = np.zeros(5)
        for i in range(len(welq)):
            if welst[i] <= day and welnd[i] > day:
                wel_data[i] = -welq[i]
                
        return [rch_data, riv_delta, wel_data], ['rch', 'riv', 'wel']
    else:
        return [rch_data, riv_delta], ['rch', 'riv']
    
    
        
    