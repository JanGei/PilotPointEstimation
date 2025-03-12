import numpy as np
from Functions.conditional_k import conditional_k
from Functions.randomK import randomK

def create_k_fields(xyz, pars: dict, k_ref, pp_xy=None, pp_cid=[], conditional=True, random=True):
    
    #%% Loading parameters
    clx = pars['lx']          # reference correlation length
    angles = pars['ang']      # reference anisotropy angle
    sigma = pars['sigma'][0]  # reference variance
    mu = pars['mu'][0]        # reference k mean
    dx = pars['dx']           # model discretization
    cxy = np.vstack((xyz[0], xyz[1])).T # reshaping (x,y) coordinates
    sig_meas = pars['sig_me'] # measurement uncertainty
    mean_range = 0.6          # range from which to draw the mean

    #%% Determining correlation initialization type
    # good initial
    if pars['covtyp'] == "good":
        lx = clx[0] # correlation lengths identical to reference 
        ang = np.random.uniform(0, np.pi) # anisotropy angle is random
        if not random:
            ang = np.deg2rad(angles[0]) # if not random - use reference
    # random initial - factor defines the upper bound of the distribution
    elif 'random' in pars['covtyp']:
        if pars['covtyp'] == 'random_low':
            factor = 0.33 
        elif pars['covtyp'] == 'random_high':
            factor = 0.50
            
        # initial correlation length drawn from [5*cell_size, factor*minor domain length]
        lx = np.array([np.random.randint(pars['dx'][0]*5, np.min(pars['nx'] * pars['dx'])*factor),
                       np.random.randint(pars['dx'][1]*5, np.min(pars['nx'] * pars['dx'])*factor)])
        ang = np.random.uniform(0, np.pi)
    
    # ensure correlation lengths are not identical 
    if lx[0] == lx[1]:
        lx[0] += 1 
    # longer correlation length should be leading
    if lx[0] < lx[1]:
        lx = np.flip(lx)
        ang = (ang +np.pi/2)%np.pi
        
    if pars['valtyp'] == 'good':
        pp_k = np.log(np.squeeze(k_ref)[pp_cid]) 
        pp_k = pp_k + sig_meas * np.random.randn(*pp_k.shape)
    else:
        if pars['valtyp'] == 'random_good':
            mu = mu
        elif pars['valtyp'] == 'random_low':
            mu = mu - 1 # decrease mean by factor of 5
        elif pars['valtyp'] == 'random_high':
            mu = mu + 1 # increase mean by factor of 5
        
        low_bound, high_bound = mu + np.array([-mean_range,mean_range])
    
    # drawing a random mean for initial field
    mean_val = np.random.uniform(low_bound, high_bound)
    # print(round(low_bound, 2), round(mean_val, 2), round(high_bound, 2))
    
    if pars['f_meas']:
        pp_loc_meas = pars['f_m_id']
        true_ppk = np.log(np.squeeze(k_ref)[pp_cid.astype(int)]) 
        pp_k_meas = true_ppk[pp_loc_meas]
        pp_k_meas = np.log(np.exp(pp_k_meas) + np.random.randn(*pp_k_meas.shape) * 0.1 * np.exp(pp_k_meas))
        pp_xy_meas = pp_xy[pp_loc_meas]
        sigma2 = np.var(pp_k_meas)
        field, field2f = conditional_k(cxy, dx, lx, ang, sigma2, pars, pp_k_meas, pp_xy_meas)
        pp_k = field[pp_cid.astype(int)]
        
        lx_iso = np.array([np.mean(lx), np.mean(lx)])
        field_iso, field2f = conditional_k(cxy, dx, lx_iso, ang, sigma2, pars, pp_k_meas, pp_xy_meas)
        pp_k_iso = field[pp_cid.astype(int)]
        
    else:
        sigma2 = np.random.uniform(0.5, 3)
        field1D = randomK(ang, sigma, pars, lx = lx, Kg = mean_val, cxy = cxy)
        pp_k = field1D[pp_cid.astype(int)]
        lx_iso = np.array([np.mean(lx), np.mean(lx)])
        field1D_iso = randomK(ang, sigma, pars, lx = lx_iso, Kg = mean_val, cxy = cxy)
        pp_k_iso = field1D_iso[pp_cid.astype(int)]

    D = pars['rotmat'](ang)
    M = np.matmul(np.matmul(D, np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]])), D.T)
    
    return field1D, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang], [pp_xy, pp_k], field1D_iso, pp_k_iso, [lx_iso, sigma2]
