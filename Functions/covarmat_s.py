import numpy as np

def covarmat_s(Xint, Xmeas, pars, theta):
    # Xint: (n x dim) array of interpolation locations
    # Xmeas: (m x dim) array of measurement locations
    # Ctype: type of covariance model (1: exponential, 2: Gaussian, 3: Matern)
    # theta: (n_theta x 1) array of structural parameters
    #        first: variance
    #        following: correlation lengths
    #        if only one corr. length, assume isotropy

    lx = np.array(theta[1]).flatten()

    Xint_rot = np.vstack(pars['rot2df'](Xint[:,0],Xint[:,1], -theta[2])).T
    Xmeas_rot = np.vstack(pars['rot2df'](Xmeas[:,0],Xmeas[:,1], -theta[2])).T

    H = pars['dstmat'](Xint_rot, Xmeas_rot, lx=lx[0], ly=lx[1])
    
    Q_ssm = pars['covmat'](H, np.array(theta[0]), pars['cov'])

    return Q_ssm