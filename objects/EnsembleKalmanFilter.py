import numpy as np

class EnsembleKalmanFilter:
    '''
    This class contians an Ensemble Kalman Filter object (Evensen, 1994),.
    
    X = [x1, x2, ... xn] is a matrix containing the model states and potentially
    the appended model parameters for all model realizations. There are n_mem 
    model realization, each contributing one vector of states and parameters
    [x1, x2, ... xn]  to X. Thus, its size is (nx , n_mem) with 
    nx = number of state + number of parameters
    
    Ysim = [ysim1, ysim2, ... ysimn] contains simulated outcomes for each model
    realization, i.e. ysim(i) = f(x(i)) with f being the model operator (e.g.
    MODFLOW). Note that ysim(i) does not correspond to the entire model state,
    but to some selected measurement locations. The number of measurement 
    locations is ny. Thus, its size is (ny , n_mem) 
    
    Cyy is the covariance matrix of the observed state, e.g. hydraulic head. 
    The formulation of the Ensemble Kalman Filter in this application allows
    the omission of the cross-covariance matrix Cxy as it is computationally
    expensive to obtain. Its size is (ny, ny)
    
    eps is the random noise component to pertrub the simulated measurements
    '''
    
    def __init__(self, X, Ysim, damp, eps, localisation):
        self.X          = X
        self.n_mem      = X.shape[1]
        self.Ysim       = Ysim
        self.eps        = eps
        self.damp       = damp
        self.X_prime    = np.zeros(np.shape(X))
        self.Y_prime    = np.zeros(np.shape(Ysim))
        self.n_obs      = np.shape(Ysim)[0]
        self.Cyy        = np.zeros((self.n_obs, self.n_obs))
        self.local      = localisation.T
        
    def update_X_Y(self, X, Y):
        self.X      = X
        self.Ysim   = Y
    
    def update_damp(self, damp_new):
        self.damp = damp_new
        
    def analysis(self):
        
        Xmean   = np.tile(np.array(np.mean(self.X, axis = 1)).T, (self.n_mem, 1)).T
        Ymean   = np.tile(np.array(np.mean(self.Ysim,  axis  = 1)).T, (self.n_mem, 1)).T
        
        # Fluctuations around mean
        X_prime = self.X - Xmean
        Y_prime = self.Ysim  - Ymean
        
        # Measurement uncertainty matrix
        R       = np.identity(self.n_obs) * self.eps**2
        
        # Covariance matrix
        Cyy     = 1/(self.n_mem-1)*np.matmul((Y_prime),(Y_prime).T) + R
        Cxy     = 1/(self.n_mem-1)*np.matmul((X_prime),(Y_prime).T)
        
        self.X_prime = X_prime
        self.Y_prime = Y_prime
        self.Cxy = Cxy
        self.Cyy = Cyy                       
    
    
    def Kalman_update(self,  Y_obs, t_step):
        Y_obs = np.tile(Y_obs, (self.n_mem,1)).T
        # perturb measurements
        Y_obs -= np.random.normal(loc=0, scale=self.eps, size=Y_obs.shape)
        # if t_step < 80:
        #     reduction = 0.01 + (1 - 0.01) / 80 * t_step
        #     self.X += reduction * self.damp[:, np.newaxis] * np.matmul(self.Cxy * self.local,
        #             np.matmul(np.linalg.inv(self.Cyy), (Y_obs - self.Ysim)))
        # else:
        self.X += self.damp[:, np.newaxis] * np.matmul(self.Cxy * self.local,
                np.matmul(np.linalg.inv(self.Cyy), (Y_obs - self.Ysim)))
        
        # self.X += 1/(self.n_mem-1) * (self.damp *
        #             np.matmul(
        #                 self.X_prime, np.matmul(
        #                     self.Y_prime.T, np.matmul(
        #                         np.linalg.inv(self.Cyy), (Y_obs - self.Ysim)
        #                         )
        #                     )
        #                 ).T
        #             ).T
        