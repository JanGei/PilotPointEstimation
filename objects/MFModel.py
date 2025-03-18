import flopy
import numpy as np
import os 
import shutil
from Functions.conditional_k import conditional_k
from Functions.Kriging import Kriging

class MFModel:
    
    def __init__(self, direc: str,  pars, obs_cid, pp_loc, l_angs = [], ellips = [], iso = False):
        self.direc      = direc
        self.mname      = pars['mname']
        self.pars       = pars
        self.iso        = iso
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
        self.cxy        = np.vstack((self.mg.xyzcellcenters[0], self.mg.xyzcellcenters[1])).T
        self.dx         = pars['dx']
        self.old_npf    = np.squeeze(self.npf.k.array)
        self.old_ic     = np.squeeze(self.ic.strt.get_data())
        self.n_neg_def  = 0
        self.pp_xy      = pp_loc[0]
        self.pp_cid     = pp_loc[1]
        self.obs_cid    = [int(i) for i in obs_cid]
        self.log        = []
        if pars['pilotp'] and not iso:
            self.ellips_mat = np.array([[ellips[0], ellips[1]], [ellips[1], ellips[2]]])
            self.lx         = [l_angs[0], l_angs[1]]
            self.ang        = l_angs[2]
            self.sigma2     = 0
            self.a          = 0.5
            self.corrL_max  = np.min(pars['nx'] * pars['dx'])
            self.threshold  = self.corrL_max * self.a
        if iso:
            self.lx         = l_angs[0]
            self.ang        = 0
            self.sigma2     = l_angs[1]
        if pars['val1st']:
            self.params     = pars['EnKF_p'][0]
        else:
            self.params     = pars['EnKF_p']
                     
                       
    def simulation(self):
        success, buff = self.sim.run_simulation()
        if not success:
            self.log.append(f'{os.path.join(*self.direc.split(os.sep)[-2:])} did not converge')
            self.set_field([self.old_npf], ['npf'])
            self.set_field([self.old_ic], ['h'])
            success, buff = self.sim.run_simulation()
            if not success:
                self.log.append(f'{os.path.join(*self.direc.split(os.sep)[-2:])} critically did not converge')      
        else:
            self.olc_ic = np.squeeze(self.get_field(['h']))
            self.old_npf = np.squeeze(self.npf.k.get_data())
     
    def copy_transient(self, packages):
        for pkg in packages:
            file = os.path.join(self.pars['trs_ws'], self.pars['mname']+'.'+pkg)
            dest_file = os.path.join(self.direc, os.path.basename(file))
            
            # Ensure the destination file is removed before copying
            if os.path.exists(dest_file):
                os.remove(dest_file)
            
            shutil.copy2(file, self.direc)

            
    def update_ic(self):
        self.ic.strt.set_data(self.get_field('h')['h'])
        self.ic.write()
       
    def Kalman_vec(self, h_mask, iso):
        
        if iso:
            data = self.get_field(['h', 'npf'])
            ysim = np.squeeze(data['h'][self.obs_cid])
            h_nobc = data['h'][~h_mask]
            x = np.concatenate([np.log(data['npf'][self.pp_cid]), h_nobc])
            
        else:
            data = self.get_field(['h', 'npf', 'cov_data'])
            
            ysim = np.squeeze(data['h'][self.obs_cid])
            h_nobc = data['h'][~h_mask]
    
            if 'cov_data' in self.params:
                if 'npf' in self.params:
                    x = np.concatenate([data['cov_data'], np.log(data['npf'][self.pp_cid]), h_nobc])
                else:
                    x = np.concatenate([data['cov_data'], h_nobc])
            else:
                if self.pars['pilotp']:
                    x = np.concatenate([np.log(data['npf'][self.pp_cid]), h_nobc])
                else:
                    x = np.concatenate([data['npf'], h_nobc])
        
        return x, ysim
    
    def updateFilter(self):
        self.params = self.pars['EnKF_p'][1]
    
    def apply_x(self, x, h_mask, mean_cov_par = [], var_cov_par = []):
        
        if self.iso:
            
            data = self.get_field(['h', 'npf'])
            data['h'][~h_mask] = x[self.pars['n_PP']:]
            
            res = self.kriging([x[:self.pars['n_PP']]])
            self.set_field([data['h']], ['h'])
        else:
            data = self.get_field(['h', 'npf', 'cov_data'])
            cl = 3
    
            if 'cov_data' in self.params:
                if 'npf' in self.params:
                    data['h'][~h_mask] = x[self.pars['n_PP']+cl:]
                    res = self.kriging([x[0:cl],x[cl:self.pars['n_PP']+cl]],
                                       mean_cov_par,
                                       var_cov_par)
                else:
                    data['h'][~h_mask] = x[cl:]
                    res = self.kriging([x[0:cl], x[cl:self.pars['n_PP']+cl]],
                                       mean_cov_par,
                                       var_cov_par)
            else:
                if self.pars['pilotp']:
                    data['h'][~h_mask] = x[self.pars['n_PP']:]
                    
                    field, _ = conditional_k(self.cxy,
                                          self.dx,
                                          self.lx,
                                          self.ang,
                                          self.pars['sigma'][0],
                                          self.pars,
                                          x[0:self.pars['n_PP']:],
                                          self.pp_xy,
                                          )
                    
                    self.set_field([field], ['npf'])
                    res = []
                else:
                    print('this is not functional yeat')
    
            
            self.set_field([data['h']], ['h'])
    
            return res
    
    def kriging(self, data, mean_cov_par = [], var_cov_par = []):
        
        if self.iso:
            pp_k = data[0]
            self.sigma2 = np.var(pp_k)
            field = conditional_k(self.cxy,
                                  self.dx,
                                  self.lx,
                                  self.ang,
                                  self.sigma2,
                                  self.pars,
                                  pp_k,
                                  self.pp_xy,
                                  )
            self.set_field([field[0]], ['npf'])
        else:
            mat, pos_def = self.check_new_matrix(data[0])
            
            if pos_def:
                l1, l2, angle = self.pars['mat2cv'](mat)    
                l1, l2, angle = self.check_vario(l1,l2, angle)
                
                pp_k = data[1]
                self.sigma2 = np.var(pp_k)
                if self.pars['condfl']:
                    field = conditional_k(self.cxy,
                                          self.dx,
                                          self.lx,
                                          self.ang,
                                          self.sigma2,
                                          self.pars,
                                          pp_k,
                                          self.pp_xy,
                                          )
                else:
                    field = Kriging(self.cxy,
                                    self.dx,
                                    self.lx,
                                    self.ang,
                                    self.pars['sigma'][0],
                                    self.pars,
                                    pp_k,
                                    self.pp_xy)
                
                self.set_field([field[0]], ['npf'])
                
            else:
                l1, l2, angle = self.pars['mat2cv'](self.ellips_mat)
                self.n_neg_def += 1 
                if self.n_neg_def == 10:
                    self.log.append(f'{os.path.join(*self.direc.split(os.sep)[-2:])} replaced cov model')
                    self.replace_model(mean_cov_par, var_cov_par)
                    
            return [[l1, l2, angle], [self.ellips_mat[0,0], self.ellips_mat[1,0], self.ellips_mat[1,1]], pos_def]
                    
    
    def update_ellips_mat(self, mat):
        self.ellips_mat = mat.copy()
     
    def replace_model(self, mean_cov_par, var_cov_par):
        pos_def = False
        while not pos_def:
            a = np.random.normal(mean_cov_par[0,0], np.sqrt(var_cov_par[0,0]))
            m = np.random.normal(mean_cov_par[0,1], np.sqrt(var_cov_par[0,1]))
            b = np.random.normal(mean_cov_par[1,1], np.sqrt(var_cov_par[1,1]))
        
            eigenvalues, eigenvectors, mat, pos_def = self.check_new_matrix([a,m,b])
        self.n_neg_def == 0    
        l1, l2, angle = self.kriging([mat[0,0], mat[1,0], mat[1,1]],
                                     mean_cov_par,
                                     var_cov_par)
    
    
    def check_new_matrix(self, data):
        
        mat = np.diag([data[0], data[2]]) + data[1] * (1 - np.eye(2))
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        
        #check for positive definiteness
        if np.all(eigenvalues > 0):
            pos_def = True
        else:
            pos_def = False
            
        if not pos_def:
            reduction = 0.96
            difmat = mat - self.ellips_mat
            while reduction > 0:
                test_mat = self.ellips_mat + reduction * difmat
                eigenvalues, eigenvectors = np.linalg.eig(test_mat)
                if np.all(eigenvalues > 0):
                    pos_def = True
                    mat = test_mat
                    break
                else:
                    reduction -= 0.05
        
        if pos_def:
            self.update_ellips_mat(mat)
            
        return mat, pos_def

    def log_correction(self, file_path):
        if len(self.log) != 0:
            g = open(file_path,'a')
            for i in range(len(self.log)):
                g.write(self.log[i])
            g.write('\n')
            g.close()
            self.log = []
                
    def reduce_corL(self, corL):
        # reducing correlation lengths based on monod kinetic model
        return (self.corrL_max * corL) / (self.corrL_max*(1-self.a) + corL)
        
    def check_vario(self, l1, l2, angle):
        correction = False
        
        # if l2 > l1:
        #     correction = True
        #     l1, l2 = l2, l1
        #     angle = angle + np.pi/2
        #     print("It happened")
            
        if l1 > self.threshold:
            l1 = self.reduce_corL(l1)
            correction = True
            
        if l2 > self.threshold:
            l2 = self.reduce_corL(l2)
            correction = True
            
        # while angle > np.pi:
        #     angle -= np.pi
        #     correction = True
        # while angle < 0:
        #     angle += np.pi
        #     correction = True
            
        if correction:
            self.variogram_to_matrix(l1, l2, angle)
        
        self.lx = [l1, l2]
        self.ang = angle
        
        return l1, l2, angle
    
    def variogram_to_matrix(self, l1, l2, angle):
        D = self.pars['rotmat'](angle)
        M = np.matmul(np.matmul(D, np.array([[1/l1**2, 0],[0, 1/l2**2]])), D.T)
        self.update_ellips_mat(M)
        
        
    def set_field(self, field, pkg_name: list):
        for i, name in enumerate(pkg_name):
            if name == 'npf':
                if self.pars['wel_k']:
                    wel = self.get_field(['wel'])['wel']
                    wel_cid = [i[-1] for i in wel[0]['cellid'][wel[0]['q'] != 0]]
                    for wel_id in wel_cid:
                        field[i][wel_id] = 1
                self.npf.k.set_data(field[i])
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
            
    def update_initial_conditions(self):
        self.sim.run_simulation()
        self.ic.strt.set_data(self.gwf.output.head().get_data())
        self.copy_transient(['rch', 'riv', 'sto', 'wel'])
        self.sim.write_simulation()
        

        
    
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
            elif name == 'cov_data':
                fields.update({name:np.array([self.ellips_mat[0,0],
                                              self.ellips_mat[0,1],
                                              self.ellips_mat[1,1]])})
            else:
                print(f'The package {name} that you requested is not part of the model')
                
        return fields
        