#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the Ensemble class. This class contains an ensemble of MF6
models and functions to controll and manipulate its members. Further, it records
the state of the ensemble 
equips it with class functions to controll and manipulate the model files

Input:
    members         ensemble members of the MFModel class
    pars            dictionary of parameters
    obs_cid         cell id's (cid) of cells with observation wells
    nprocs          number of processors for parallel computing
    mask            mask of cells with 0 variance in hydraulic head
    k_ref           reference hydraulic conductivity field
    ellipses        covariance function parameters of all members (correlation lengths, angels)
    ellipses_par    parameteric covariance function paramters in ellipse representation
    pp_cid          cell id's of pilot points
    pp_xy           cell xy coordiates of pilot points
    pp_k            hydraulic conductivity values at pilot points
    iso             flag whether ensemble is isotropic or anisotropic
    
@author: janek geiger
"""
from joblib import Parallel, delayed
import numpy as np
import os
    
class Ensemble:
    def __init__(self, members: list, pars, obs_cid, nprocs: int, mask, k_ref, ellipses = [], ellipses_par = [], pp_cid = [], pp_xy = [], pp_k = [], iso = False):
        '''
        

        Parameters
        ----------
        members : list
            DESCRIPTION.
        pars : TYPE
            DESCRIPTION.
        obs_cid : TYPE
            DESCRIPTION.
        nprocs : int
            DESCRIPTION.
        mask : TYPE
            DESCRIPTION.
        k_ref : TYPE
            DESCRIPTION.
        ellipses : TYPE, optional
            DESCRIPTION. The default is [].
        ellipses_par : TYPE, optional
            DESCRIPTION. The default is [].
        pp_cid : TYPE, optional
            DESCRIPTION. The default is [].
        pp_xy : TYPE, optional
            DESCRIPTION. The default is [].
        pp_k : TYPE, optional
            DESCRIPTION. The default is [].
        iso : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        self.members    = members
        self.pars       = pars
        self.nprocs     = nprocs
        self.n_mem      = len(self.members)
        self.h_mask     = mask.astype(bool)
        self.ole        = {'assimilation': [], 'prediction': []}
        self.ole_nsq    = {'assimilation': [], 'prediction': []}
        self.te1        = {'assimilation': [], 'prediction': []}
        self.te1_nsq    = {'assimilation': [], 'prediction': []}
        self.te2        = {'assimilation': [], 'prediction': []}
        self.te2_nsq    = {'assimilation': [], 'prediction': []}
        self.nrmse      = {'assimilation': [], 'prediction': []}
        self.nrmse_nsq  = {'assimilation': [], 'prediction': []}
        self.te1_k      = {'normal': [], 'nsq': []}
        self.te2_k      = {'normal': [], 'nsq': []}
        self.nrmse_k    = {'normal': [], 'nsq': []}
        self.k_ref      = k_ref
        self.k_ref_log  = np.log(k_ref)
        self.obs        = []
        self.pilotp_flag= pars['pilotp']
        self.obs_cid    = [int(i) for i in obs_cid]
        self.meanlogk   = []
        self.meank      = []
        self.iso        = iso
        self.vark       = []
        if pars['pilotp']:
            self.ellipses   = ellipses
            self.ellipses_par = ellipses_par
            self.pp_cid     = pp_cid
            self.pp_xy      = pp_xy
            self.mean_cov   = np.mean(ellipses, axis = 0)
            self.var_cov    = np.var(ellipses, axis = 0)
            self.mean_cov_par   = np.mean(ellipses_par, axis = 0)
            self.var_cov_par    = np.var(ellipses_par, axis = 0)
            self.meanlogppk = []
            self.varlogppk  = []
            self.meanppk    = []
            self.varppk     = []
            self.pp_k_ini   = pp_k
        if pars['val1st']:
            self.params     = pars['EnKF_p'][0]
        else:
            self.params     = pars['EnKF_p']
        
        
    def set_field(self, field, pkg_name: list):
        Parallel(n_jobs=self.nprocs, backend=self.pars['backnd'])(delayed(self.members[idx].set_field)(
            [field[idx]],
            pkg_name) 
            for idx in range(self.n_mem)
            )
    
    def update_initial_conditions(self):
        Parallel(n_jobs=self.nprocs,
                 backend=self.pars['backnd'])(
                     delayed(self.members[idx].update_initial_conditions)(
                         ) 
                     for idx in range(self.n_mem)
                     )
        
        
    def propagate(self):
        Parallel(n_jobs=self.nprocs, backend=self.pars['backnd'])(delayed(self.members[idx].simulation)(
            ) 
            for idx in range(self.n_mem)
            )
        
    def update_initial_heads(self):
        Parallel(n_jobs=self.nprocs, backend=self.pars['backnd'])(delayed(self.members[idx].update_ic)(
            ) 
            for idx in range(self.n_mem)
            )
        
    def get_damp(self, X, switch = False):
        
        if self.pars['val1st']:
            if switch:
                self.params = self.pars['EnKF_p'][1]
                val = self.pars['damp'][1]
                self.updateFilter()
            else:
                val = self.pars['damp'][0]

        else:
            val = self.pars['damp']
            
        damp = np.zeros((X[:,0].size)) + val[0]
        
        if self.iso:
            damp[:len(self.pp_cid)] = val[2]
            damp[len(self.pp_cid):] = val[0]
            if self.pars['f_meas']:
                damp[self.pars['f_m_id']] = val[2] / 50
        else:
            if 'cov_data' in self.params:
                cl = len(np.unique(self.members[0].ellips_mat))
                damp[0], damp[2] = val[1][0], val[1][0]
                damp[1] = val[1][1]
                if 'npf' in self.params:
                    damp[cl:cl+len(self.pp_cid)] = val[2]
                    
                    if self.pars['f_meas']:
                        ids = cl +self.pars['f_m_id']
                        damp[ids] = val[2] / 50
            else:
                if self.pilotp_flag:
                    damp[:len(self.pp_cid)] = val[1]
                else:
                    damp[:len(self.members[0].npf.k.array.squeeze())] = val[1]
            
        return damp
        
    def write_simulations(self):
        Parallel(n_jobs=self.nprocs,
                 backend=self.pars['backnd'])(
                     delayed(self.members[idx].write_sim)()
                     for idx in range(self.n_mem)
                     )
                     
    def updateFilter(self):
        Parallel(n_jobs=self.nprocs,
                 backend=self.pars['backnd'])(
                     delayed(self.members[idx].updateFilter)() 
            for idx in range(self.n_mem)
            )
        
    def apply_X(self, X):
        
        if self.iso:
            Parallel(n_jobs=self.nprocs, backend=self.pars['backnd'])(delayed(self.members[idx].apply_x)(
                np.squeeze(X[:,idx]),
                self.h_mask,
                ) 
                for idx in range(self.n_mem)
                )
            
            self.meanlogppk = np.mean(X[:len(self.pp_cid),:], axis = 1)
            self.varlogppk = np.var(X[:len(self.pp_cid),:], axis = 1)
            self.meanppk = np.mean(np.exp(X[:len(self.pp_cid),:]), axis = 1)
            self.varppk = np.var(np.exp(X[:len(self.pp_cid),:]), axis = 1)
            
        else:
            result = Parallel(n_jobs=self.nprocs, backend=self.pars['backnd'])(delayed(self.members[idx].apply_x)(
                np.squeeze(X[:,idx]),
                self.h_mask,
                self.mean_cov_par,
                self.var_cov_par
                ) 
                for idx in range(self.n_mem)
                )
            
            cl = 3
            if 'cov_data' in self.params:
                # Only register ellipses that perfromed a successfull update
                self.ellipses = np.array([data[0] for data in result if data[2]])
                self.ellipses_par = [data[1] for data in result if data[2]]
                # self.mean_cov = np.mean(self.ellipses, axis = 0)
                self.var_cov = np.var(self.ellipses, axis = 0)
                self.mean_cov_par = np.mean(np.array(self.ellipses_par), axis = 0)
                self.var_cov_par = np.var(np.array(self.ellipses_par), axis = 0)
                if 'npf' in self.params:
                    self.meanlogppk = np.mean(X[cl:len(self.pp_cid)+cl,:], axis = 1)
                    self.varlogppk = np.var(X[cl:len(self.pp_cid)+cl,:], axis = 1)
                    self.meanppk = np.mean(np.exp(X[cl:len(self.pp_cid)+cl,:]), axis = 1)
                    self.varppk = np.var(np.exp(X[cl:len(self.pp_cid)+cl,:]), axis = 1)
            else:
                if self.pilotp_flag:
                    self.meanlogppk = np.mean(X[:len(self.pp_cid),:], axis = 1)
                    self.varlogppk = np.var(X[:len(self.pp_cid),:], axis = 1)
                    self.meanppk = np.mean(np.exp(X[:len(self.pp_cid),:]), axis = 1)
                    self.varppk = np.var(np.exp(X[:len(self.pp_cid),:]), axis = 1)
                    
    def get_Kalman_X_Y(self):   

        result = Parallel(n_jobs=self.nprocs, backend=self.pars['backnd'])(delayed(self.members[idx].Kalman_vec)(
            self.h_mask,
            self.iso
            ) 
            for idx in range(self.n_mem)
            )
        
        xs = []
        ysims = []
        for tup in result:
            xs.append(tup[0])
            ysims.append(tup[1])
        
        X = np.vstack(xs).T
        Ysim = np.vstack(ysims).T
        
        return X, Ysim
    
    def update_transient_data(self,packages):

        Parallel(n_jobs=self.nprocs, backend=self.pars['backnd'])(delayed(self.members[idx].copy_transient)(
            packages
            ) 
            for idx in range(self.n_mem)
            )


    
    def model_error(self,  true_h, period):
        
        mean_h, var_h = self.get_mean_var(h = 'ic')
        true_h = np.squeeze(true_h)
        
        mean_obs = mean_h[self.obs_cid]
        true_obs = true_h[self.obs_cid]
        self.obs = [true_obs, mean_obs]
        
        # calculating nrmse without root for later summation
        true_h_m = true_h[~self.h_mask]
        mean_h_m = mean_h[~self.h_mask]
        var_h_m = var_h[~self.h_mask]
        var_ole = 0.01**2
        # Erdal Formulation
        # var_te2 = ((true_h_m + mean_h_m)/2)**2
        # New Formulation
        var_te2 = true_h_m**2

        # Computing normalized squared error only considering nodes
        node_ole = np.mean((true_obs - mean_obs)**2/var_ole)
        node_te1 = np.mean((true_h_m - mean_h_m)**2/var_h_m)
        node_te2 = np.mean((true_h_m - mean_h_m)**2/var_te2)
        node_nrmse = np.mean((true_h_m - mean_h_m)**2/var_ole)
        
        # Append node error to list
        self.ole_nsq[period].append(node_ole)
        self.te1_nsq[period].append(node_te1)
        self.te2_nsq[period].append(node_te2)
        self.nrmse_nsq[period].append(node_nrmse)

        # Calculate NRMSE over all node calculations
        time_ole = np.sqrt(np.mean(self.ole_nsq[period]))
        time_te1 = np.sqrt(np.mean(self.te1_nsq[period]))
        time_te2 = np.sqrt(np.mean(self.te2_nsq[period]))
        time_nrmse = np.sqrt(np.mean(self.nrmse_nsq[period]))

        # Append error to resulting list
        self.ole[period].append(time_ole)
        self.te1[period].append(time_te1)
        self.te2[period].append(time_te2)
        self.nrmse[period].append(time_nrmse)
        
        if period == 'assimilation':
            mean_k, var_k = self.get_mean_var(h = 'npf', log = True)
            
            var_te2_k = self.k_ref_log**2
            node_te1_k = np.mean((self.k_ref_log - mean_k)**2/var_k)
            node_te2_k = np.mean((self.k_ref_log - mean_k)**2/var_te2_k)
            node_nrmse_k = np.mean((self.k_ref_log - mean_k)**2/0.01**2)
            
            self.te1_k['nsq'].append(node_te1_k)
            self.te2_k['nsq'].append(node_te2_k)
            self.nrmse_k['nsq'].append(node_nrmse_k)
            
            self.te1_k['normal'].append(np.sqrt(np.mean(self.te1_k['nsq'])))
            self.te2_k['normal'].append(np.sqrt(np.mean(self.te2_k['nsq'])))
            self.nrmse_k['normal'].append(np.sqrt(np.mean(self.nrmse_k['nsq'])))

        return mean_h, var_h
    
    def get_member_fields(self, params):
        
        data = Parallel(n_jobs=self.nprocs, backend=self.pars['backnd'])(delayed(self.members[idx].get_field)(
            params
            ) 
            for idx in range(self.n_mem)
            )
        
        return data
    
    def log(self, time_step):
       
        if not self.iso:
            g = open(self.pars['logfil'],'a')
            g.write(f'Time Step {time_step}')
            g.write('\n')
            g.close()
        for member in self.members:
            member.log_correction(self.pars['logfil'])

    def get_mean_var(self, h = 'h', log = False):
        h_fields = self.get_member_fields([h])
        
        h_f = np.array([np.squeeze(field[h]) for field in h_fields]).T
        
        if h == 'npf' and log:
            h_f = np.log(h_f)

        return np.mean(h_f, axis = 1), np.var(h_f, axis = 1)
    
    def record_state(self, pars: dict, true_h, period: str, t_step):
        
        mean_h, var_h = self.get_mean_var(h = 'ic')
        k_fields = self.get_member_fields(['npf'])
        k_fields = np.array([field['npf'] for field in k_fields]).squeeze()
        self.meanlogk = np.mean(np.log(k_fields), axis = 0)
        self.varlogk = np.var(np.log(k_fields), axis = 0)
        self.meank = np.mean(k_fields, axis = 0)
        
        direc = pars['resdir']
        
        if self.iso:
            suffix = '_iso'
        else:
            suffix = ''

        f = open(os.path.join(direc,  'errors_'+period+suffix+'.dat'),'a')
        g = open(os.path.join(direc,  'errors_'+period+suffix+'_ts.dat'),'a')
        f.write("{:.3f} ".format(self.ole[period][-1]))
        f.write("{:.3f} ".format(self.te1[period][-1]))
        f.write("{:.7f} ".format(self.te2[period][-1]))
        f.write("{:.5f} ".format(self.nrmse[period][-1]))
        g.write("{:.3f} ".format(self.ole_nsq[period][-1]))
        g.write("{:.3f} ".format(self.te1_nsq[period][-1]))
        g.write("{:.7f} ".format(self.te2_nsq[period][-1]))
        g.write("{:.5f} ".format(self.nrmse_nsq[period][-1]))
        f.write('\n')
        g.write('\n')
        f.close()
        g.close()
        
        f = open(os.path.join(direc,  'errors_k'+suffix+'.dat'),'a')
        g = open(os.path.join(direc,  'errors_k_ts'+suffix+'.dat'),'a')
        f.write("{:.3f} ".format(self.te1_k['normal'][-1]))
        f.write("{:.7f} ".format(self.te2_k['normal'][-1]))
        f.write("{:.5f} ".format(self.nrmse_k['normal'][-1]))
        g.write("{:.3f} ".format(self.te1_k['nsq'][-1]))
        g.write("{:.7f} ".format(self.te2_k['nsq'][-1]))
        g.write("{:.5f} ".format(self.nrmse_k['nsq'][-1]))
        f.write('\n')
        g.write('\n')
        f.close()
        g.close()
        
        f = open(os.path.join(direc,  'obs_true'+suffix+'.dat'),'a')
        g = open(os.path.join(direc,  'obs_mean'+suffix+'.dat'),'a')
        for i in range(len(self.obs[0])):
            f.write("{:.2f} ".format(self.obs[0][i]))
            g.write("{:.2f} ".format(self.obs[1][i]))
        f.write('\n')
        g.write('\n')
        f.close()
        g.close()
        
        if t_step%20 == 0:
            f = open(os.path.join(direc,  'h_mean'+suffix+'.dat'),'a')
            g = open(os.path.join(direc,  'h_var'+suffix+'.dat'),'a')
            h = open(os.path.join(direc,  'true_h'+suffix+'.dat'),'a')
            for i in range(len(mean_h)):
                f.write("{:.2f} ".format(mean_h[i]))
                g.write("{:.2f} ".format(var_h[i]))
                h.write("{:.2f} ".format(true_h[i]))
            f.write('\n')
            g.write('\n')
            h.write('\n')
            f.close()
            g.close()
            h.close()

        
        # also store covariance data for all models
        if not self.iso:
            cov_data = self.get_member_fields(['cov_data'])
            
            mat = np.array([[self.mean_cov_par[0], self.mean_cov_par[1]],
                            [self.mean_cov_par[1], self.mean_cov_par[2]]])
            res = pars['mat2cv'](mat)
                
            f = open(os.path.join(direc, 'covariance_data.dat'),'a')
            f.write("{:.2f} ".format(res[0]))
            f.write("{:.2f} ".format(res[1]))
            f.write("{:.2f} ".format(res[2]))
            f.write('\n')
            f.close()
            
            f = open(os.path.join(direc, 'cov_variance.dat'),'a')
            f.write("{:.2f} ".format(self.var_cov[0]))
            f.write("{:.2f} ".format(self.var_cov[1]))
            f.write("{:.4f} ".format(self.var_cov[2]))
            f.write('\n')
            f.close()
            
            f = open(os.path.join(direc, 'covariance_data_par.dat'),'a')
            f.write("{:.10f} ".format(self.mean_cov_par[0]))
            f.write("{:.10f} ".format(self.mean_cov_par[1]))
            f.write("{:.10f} ".format(self.mean_cov_par[2]))
            f.write('\n')
            f.close()
        
            
            f = open(os.path.join(direc, 'cov_variance_par.dat'),'a')
            f.write("{:.2f} ".format(np.log(self.var_cov_par[0])))
            f.write("{:.2} ".format(np.log(self.var_cov_par[1])))
            f.write("{:.2f} ".format(np.log(self.var_cov_par[2])))
            f.write('\n')
            f.close()
            
            for i in range(self.n_mem):
                f = open(os.path.join(direc, f'covariance_model_{i}.dat'), 'a')
                for j in range(len(cov_data[i]['cov_data'])):
                    f.write("{:.10f} ".format(cov_data[i]['cov_data'][j]))
                f.write('\n')
                f.close()

        if 'npf' in self.params:
            if self.pilotp_flag:
                f = open(os.path.join(direc,  'meanlogppk'+suffix+'.dat'),'a')
                g = open(os.path.join(direc,  'varlogppk'+suffix+'.dat'),'a')
                for i in range(len(self.meanlogppk)):
                    f.write("{:.2f} ".format(self.meanlogppk[i]))
                    g.write("{:.2f} ".format(self.varlogppk[i]))
                f.write('\n')
                g.write('\n')
                f.close()
                g.close()
                
                f = open(os.path.join(direc,  'meanppk'+suffix+'.dat'),'a')
                g = open(os.path.join(direc,  'varppk'+suffix+'.dat'),'a')
                for i in range(len(self.meanppk)):
                    f.write("{:.5f} ".format(self.meanppk[i]))
                    g.write("{:.5f} ".format(self.varppk[i]))
                f.write('\n')
                g.write('\n')
                f.close()
                g.close()
            
            if t_step%20 == 0:
                f = open(os.path.join(direc,  'meanlogk'+suffix+'.dat'),'a')
                g = open(os.path.join(direc,  'varlogk'+suffix+'.dat'),'a')
                for i in range(len(self.meanlogk)):
                    f.write("{:.2f} ".format(self.meanlogk[i]))
                    g.write("{:.2f} ".format(self.varlogk[i]))
                f.write('\n')
                g.write('\n')
                f.close()
                g.close()
            
                f = open(os.path.join(direc,  'meank'+suffix+'.dat'),'a')
                for i in range(len(self.meank)):
                    f.write("{:.5f} ".format(self.meank[i]))
                f.write('\n')
                f.close()
                
    def record_shadow_state(self, pars: dict, true_h, period: str, t_step):
        
        mean_h, var_h = self.get_mean_var(h = 'ic')
        k_fields = self.get_member_fields(['npf'])
        k_fields = np.array([field['npf'] for field in k_fields]).squeeze()
        self.meanlogk = np.mean(np.log(k_fields), axis = 0)
        self.varlogk = np.var(np.log(k_fields), axis = 0)
        self.meank = np.mean(k_fields, axis = 0)
        
        direc = pars['resdir']

        f = open(os.path.join(direc,  'errors_'+period+'shadow.dat'),'a')
        g = open(os.path.join(direc,  'errors_'+period+'shadow_ts.dat'),'a')
        f.write("{:.3f} ".format(self.ole[period][-1]))
        f.write("{:.3f} ".format(self.te1[period][-1]))
        f.write("{:.7f} ".format(self.te2[period][-1]))
        f.write("{:.5f} ".format(self.nrmse[period][-1]))
        g.write("{:.3f} ".format(self.ole_nsq[period][-1]))
        g.write("{:.3f} ".format(self.te1_nsq[period][-1]))
        g.write("{:.7f} ".format(self.te2_nsq[period][-1]))
        g.write("{:.5f} ".format(self.nrmse_nsq[period][-1]))
        f.write('\n')
        g.write('\n')
        f.close()
        g.close()
        
        f = open(os.path.join(direc,  'errors_k_shadow.dat'),'a')
        g = open(os.path.join(direc,  'errors_k_ts_shadow.dat'),'a')
        f.write("{:.3f} ".format(self.te1_k['normal'][-1]))
        f.write("{:.7f} ".format(self.te2_k['normal'][-1]))
        f.write("{:.5f} ".format(self.nrmse_k['normal'][-1]))
        g.write("{:.3f} ".format(self.te1_k['nsq'][-1]))
        g.write("{:.7f} ".format(self.te2_k['nsq'][-1]))
        g.write("{:.5f} ".format(self.nrmse_k['nsq'][-1]))
        f.write('\n')
        g.write('\n')
        f.close()
        g.close()
        
        g = open(os.path.join(direc,  'obs_mean_shadow.dat'),'a')
        for i in range(len(self.obs[0])):

            g.write("{:.2f} ".format(self.obs[1][i]))
        g.write('\n')
        g.close()
        
        if t_step%20 == 0:
            f = open(os.path.join(direc,  'h_mean_shadow.dat'),'a')
            g = open(os.path.join(direc,  'h_var_shadow.dat'),'a')
            for i in range(len(mean_h)):
                f.write("{:.2f} ".format(mean_h[i]))
                g.write("{:.2f} ".format(var_h[i]))
            f.write('\n')
            g.write('\n')
            f.close()
            g.close()
            
        if t_step == 0:
            f = open(os.path.join(direc,  'meanlogk.dat'),'a')
            g = open(os.path.join(direc,  'varlogk.dat'),'a')
            for i in range(len(self.meanlogk)):
                f.write("{:.2f} ".format(self.meanlogk[i]))
                g.write("{:.2f} ".format(self.varlogk[i]))
            f.write('\n')
            g.write('\n')
            f.close()
            g.close()
                
            

        
    def remove_current_files(self, pars):
        
        for filename in os.listdir(pars['resdir']):
            file_path = os.path.join(pars['resdir'], filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.remove(file_path)

        
    def reset_errors(self):
        self.ole        = {'assimilation': [], 'prediction': []}
        self.ole_nsq    = {'assimilation': [], 'prediction': []}
        self.te1        = {'assimilation': [], 'prediction': []}
        self.te1_nsq    = {'assimilation': [], 'prediction': []}
        self.te2        = {'assimilation': [], 'prediction': []}
        self.te2_nsq    = {'assimilation': [], 'prediction': []}
        
        
        
        
        
        
        
        
        
        