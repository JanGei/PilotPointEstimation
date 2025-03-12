import flopy
import numpy as np

class Virtual_Reality:
    
    def __init__(self,  pars, obs_cid):
        self.direc      = pars['trs_ws']
        self.mname      = pars['mname']
        self.pars       = pars
        self.sim        = flopy.mf6.modflow.MFSimulation.load(
                                version             = 'mf6', 
                                exe_name            = 'mf6',
                                sim_ws              = pars['trs_ws'], 
                                verbosity_level     = 0
                                )
        self.gwf        = self.sim.get_model(self.mname)
        self.npf        = self.gwf.npf
        self.rch        = self.gwf.rch
        self.riv        = self.gwf.riv
        self.wel        = self.gwf.wel
        self.ic         = self.gwf.ic
        self.chd        = self.gwf.chd
        self.mg         = self.gwf.modelgrid
        self.obs_cid    = [int(i) for i in obs_cid]
        self.cxy        = np.vstack((self.mg.xyzcellcenters[0], self.mg.xyzcellcenters[1])).T
        self.dx         = pars['dx']

            
        
    def set_field(self, field, pkg_name: list):
        for i, name in enumerate(pkg_name):
            if name == 'npf':
                self.old_npf =  self.npf.k.get_data()
                self.npf.k.set_data(np.reshape(field[i],self.npf.k.array.shape))
                self.npf.write()
            elif name == 'rch':
                self.rch.stress_period_data.set_data(field[i])
                self.rch.write()
            elif name == 'riv':
                self.riv.stress_period_data.set_data(field[i])
                self.riv.write()
            elif name == 'wel':
                self.wel.stress_period_data.set_data(field[i])
                self.wel.write()
            elif name == 'h':
                self.ic.strt.set_data(field[i])
                self.ic.write()
            else:
                print(f'The package {name} that you requested is not part of the model')
            
        
    def get_field(self, pkg_name: list) -> dict:
        fields = {}
        for name in pkg_name:
            if name == 'npf':
                fields.update({name:self.npf.k.get_data()})
            elif name == 'rch':
                fields.update({name:self.rch.stress_period_data.get_data()})
            elif name == 'riv':
                fields.update({name:self.riv.stress_period_data.get_data()})
            elif name == 'wel':
                fields.update({name:self.wel.stress_period_data.get_data()})
            elif name == 'chd':
                fields.update({name:self.chd.stress_period_data.get_data()})
            elif name == 'h':
                fields.update({name:self.gwf.output.head().get_data()})
            else:
                print(f'The package {name} that you requested is not part of the model')
                
        return fields
    
    def update_transient_data(self, data, packages):

        spds = self.get_field(packages)
        rch_spd = spds['rch']
        riv_spd = spds['riv']
        
        rch_spd[0]['recharge'] = data[0]
        riv_spd[0]['stage'] += data[1]
        
        if 'wel' in packages:
            wel_spd = spds['wel']
            wel_spd[0]['q'] = data[2]
            spds = [rch_spd, riv_spd, wel_spd]
        else:
            spds = [rch_spd, riv_spd]
        
        self.set_field(spds, packages) 
        
    def simulation(self):
        success, buff = self.sim.run_simulation()
        if not success:
            import sys
            print('The Virtual Reality did crash - Aborting')
            sys.exit()
        
       
    def update_ic(self):
        h_field = self.get_field('h')['h']
        self.ic.strt.set_data(h_field)
        self.ic.write()
        return h_field
    
    def get_observations(self):
        data = np.squeeze(self.get_field(['h'])['h'])
        ysim = data[self.obs_cid]
        return ysim