import flopy

def load_template_model(pars: dict,  SS = False):
    
    sim        = flopy.mf6.modflow.MFSimulation.load(
                            version             = 'mf6', 
                            exe_name            = 'mf6',
                            sim_ws              = pars['sim_ws'], 
                            verbosity_level     = 0
                            )
    gwf = sim.get_model(pars['mname'])
    
    return sim, gwf