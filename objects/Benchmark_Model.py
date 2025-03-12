import flopy
import numpy as np
import sys
import os 
import shutil


sys.path.append('..')

class B_Model:
    
    def __init__(self, direc: str,  pars, obs_cid, mask):
        self.direc      = direc
        self.mname      = pars['mname']
        self.pars       = pars
        self.sim        = flopy.mf6.modflow.MFSimulation.load(
                                version             = 'mf6', 
                                exe_name            = 'mf6',
                                sim_ws              = direc, 
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
        self.ole        = {'assimilation': [], 'prediction': []}
        self.ole_nsq    = {'assimilation': [], 'prediction': []}
        self.te2        = {'assimilation': [], 'prediction': []}
        self.te2_nsq    = {'assimilation': [], 'prediction': []}
        self.obs        = []
        self.cxy        = np.vstack((self.mg.xyzcellcenters[0], self.mg.xyzcellcenters[1])).T
        self.obs_cid    = [int(i) for i in obs_cid]
        self.h_mask     = mask.astype(bool)
                     
                       
    def simulation(self):
        success, buff = self.sim.run_simulation()
        self.update_ic()
    
    
    def model_error(self,  true_h, period):
        
        h_bench = np.squeeze(self.get_field('h')['h'])
        true_h = np.squeeze(true_h)
        
        mean_obs = h_bench[self.obs_cid]
        true_obs = true_h[self.obs_cid]
        self.obs = [true_obs, mean_obs]
        
        # calculating nrmse without root for later summation
        true_h_m = true_h[~self.h_mask]
        mean_h_m = h_bench[~self.h_mask]
        var_te2 = (true_h_m + mean_h_m)/2
        
        
        # Computing normalized squared error only considering nodes
        node_ole = np.mean((true_obs - mean_obs)**2/(0.01**2))
        node_te2 = np.mean((true_h_m - mean_h_m)**2/(var_te2**2))
        
        # Append node error to list
        self.ole_nsq[period].append(node_ole)
        self.te2_nsq[period].append(node_te2)

        # Calculate NRMSE over all node calculations
        time_ole = np.sqrt(np.mean(self.ole_nsq[period]))
        time_te2 = np.sqrt(np.mean(self.te2_nsq[period]))

        # Append error to resulting list
        self.ole[period].append(time_ole)
        self.te2[period].append(time_te2)
        
        self.record_state(self.pars, period)
        
    def record_state(self, pars: dict, period):
         
        direc = pars['resdir']
        
        f = open(os.path.join(direc,  'errors_'+period+'_benchmark.dat'),'a')
        f.write("{:.3f} ".format(self.ole[period][-1]))
        f.write("{:.6f} ".format(self.te2[period][-1]))
        f.write('\n')
        f.close()
        
        
    def copy_transient(self, packages):
        for pkg in packages:
            file = os.path.join(self.pars['trs_ws'], self.pars['mname']+'.'+pkg)
            shutil.copy(file, self.direc)

            
    def update_ic(self):
        self.ic.strt.set_data(self.get_field('h')['h'])
        self.ic.write()
         
        
    def set_field(self, field, pkg_name: list):
        for i, name in enumerate(pkg_name):
            if name == 'npf':
                self.old_npf =  self.npf.k.get_data()
                if self.pars['wel_k']:
                    wel = self.get_field(['wel'])['wel']
                    wel_cid = [i[-1] for i in wel[0]['cellid'][wel[0]['q'] != 0]]
                    field[i][wel_cid] = 1
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
                fields.update({name:np.squeeze(self.npf.k.get_data())})
            elif name == 'rch':
                fields.update({name:self.rch.stress_period_data.get_data()})
            elif name == 'riv':
                fields.update({name:self.riv.stress_period_data.get_data()})
            elif name == 'wel':
                fields.update({name:self.wel.stress_period_data.get_data()})
            elif name == 'chd':
                fields.update({name:self.chd.stress_period_data.get_data()})
            elif name == 'h':
                fields.update({name:np.squeeze(self.gwf.output.head().get_data())})
            elif name == 'ic':
                fields.update({name:np.squeeze(self.ic.strt.get_data())})
            else:
                print(f'The package {name} that you requested is not part of the model')
                
        return fields
        
    def reset_errors(self):
        self.ole        = {'assimilation': [], 'prediction': []}
        self.ole_nsq    = {'assimilation': [], 'prediction': []}
        self.te1        = {'assimilation': [], 'prediction': []}
        self.te1_nsq    = {'assimilation': [], 'prediction': []}
        self.te2        = {'assimilation': [], 'prediction': []}
        self.te2_nsq    = {'assimilation': [], 'prediction': []}