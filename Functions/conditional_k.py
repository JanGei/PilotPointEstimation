from randomK import randomK
from covarmat_s import covarmat_s
import numpy as np
from scipy.interpolate import griddata


def conditional_k(cxy, dx, lx, ang, sigma, pars, pp_k, pp_xy): 
    
    # Associated Grid locations with 0,0 in the left bottom corner
    xint = np.arange(dx[0]/2, pars['nx'][0]*dx[0] + dx[0]/2, dx[0])
    yint = np.arange(dx[1]/2, pars['nx'][1]*dx[1] + dx[1]/2, dx[1])

    # Grid in Physical Coordinates
    Xint, Yint = np.meshgrid(xint, yint)
    # Xint_pw = np.column_stack((Xint.T.ravel(), Yint.T.ravel()))
    Xint_pw = np.column_stack((Xint.ravel(order = 'F'), Yint.ravel(order = 'F')))
    
    # Construct covariance matrix of measurement error
    m = len(pp_k)
    n = Xint_pw.shape[0]
    R = np.eye(m)* pars['sig_me']**2
    # R = np.eye(m)* 0.1**2
    
    # Discretized trend functions (for constant mean)
    X = np.ones((n,1))
    Xm = np.ones((m,1))
    
    # Construct the necessary covariance matrices
    Qssm = covarmat_s(Xint_pw,pp_xy,pars,[sigma,lx,ang]) 
    Qsmsm = covarmat_s(pp_xy,pp_xy,pars,[sigma,lx, ang])
        
    # kriging matrix and its inverse
    krigmat = np.vstack((np.hstack((Qsmsm+R, Xm)), np.append(Xm.T, 0)))
    # ikrigmat = np.linalg.inv(krigmat)
    
    # random, unconditional field for the given variogram
    sunc = np.log(randomK(ang, sigma, pars, lx = lx))
    
    # generating a conditional realisation
    sunc_at_meas = np.zeros(m)
    indmeas = np.floor(pp_xy @ np.linalg.inv(np.diag(dx))) # indices of the grid
    for ii in range(m):
        sunc_at_meas[ii] = sunc[int(indmeas[ii,1]),
                                int(indmeas[ii,0])] 
    
    # Perturb the measurements and subtract the unconditional realization
    spert = np.squeeze(pp_k)- np.squeeze(sunc_at_meas) #+ np.squeeze(0.1 * np.random.randn(*pp_k.shape)) 
    
    # Solve the kriging equation
    sol = np.linalg.solve(krigmat, np.append(spert, 0))
    
    # Separate the trend coefficient(s) from the weights of the covariance-functions in the function-estimate form
    xi = sol[:m]
    beta = sol[m]

    s_cond = np.squeeze(Qssm @ xi.reshape((len(xi),1))) + np.squeeze(X * beta) + sunc.flatten(order = 'F')
    s_cond_grid = np.reshape(s_cond,Xint.shape, order='F')     

    # generating an associated grid for K 
    values_at_coordinates = griddata((Xint.ravel(order = 'F'), Yint.ravel(order = 'F')), s_cond.ravel(order = 'F'),
                                     (cxy[:,0], cxy[:,1]), method='nearest')
    

    return np.exp(values_at_coordinates), np.exp(s_cond_grid)