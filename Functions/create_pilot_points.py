from scipy.stats import qmc
import numpy as np
import flopy
from shapely.geometry import MultiPoint, Point

    
def create_pilot_points(gwf, pars:dict,):
    '''
    Generate randomly spaced pilot points

    Parameters
    ----------
    gwf : flopy object
        Groundwater flow object (MF6 model).
    pars : dict
        Simulation / model parameters.

    Returns
    -------
    arrays of float
        Cell ID's and xy coordinates of pilot points.

    '''
    # Save old rng state
    rng_state = np.random.get_state()
    # Fix random seed for reproducibility
    np.random.seed(805)
    
    nPP = pars['n_PP']
    mg = gwf.modelgrid
    ixs  = flopy.utils.GridIntersect(mg, method = "vertex")
    vert = mg.xyzvertices
    xmax = np.max([np.max(list) for list in vert[0]])
    ymax = np.max([np.max(list) for list in vert[1]])
    xyz = mg.xyzcellcenters
    xy = list(zip(xyz[0], xyz[1]))
    
    welxy = pars['welxy']
    obsxy = pars['obsxy']
    
    welcid = ixs.intersect(MultiPoint(welxy))
    obscid = ixs.intersect(MultiPoint(obsxy))
    
    blocked_cid = np.concatenate((welcid.cellids,obscid.cellids))
    pp_cid_accepted = []
    pp_xy_accepted = []
    
    n_test = nPP
    if pars['scramb']:
        np.random.seed(805)
        
    sampler = qmc.Halton(2, seed = 60)
    
    while len(pp_cid_accepted) != nPP:
        # Pilot points should not end up on observationand pumping wells
        pp_xy_proposal = (sampler.random(n = n_test) * np.array([xmax, ymax]))
        
        pp_cid_proposal = np.zeros(len(pp_xy_proposal))
        for i, point in  enumerate(pp_xy_proposal):
            pp_cid_proposal[i] = ixs.intersect(Point(point)).cellids.astype(int)[0]
        
        common_cells = np.intersect1d(pp_cid_proposal, blocked_cid)
        
        pp_cid_accepted = np.setdiff1d(pp_cid_proposal, common_cells)
        
        # get xy coordinated from proposed cells
        if len(pp_cid_accepted) == nPP:
            pp_xy_accepted = np.array([xy[int(i)] for i in pp_cid_accepted])
        
        if len(pp_cid_accepted) > nPP:
            n_test -= int((len(pp_cid_proposal) - nPP) /2)
        elif len(pp_cid_accepted) < nPP:
            n_test += int((nPP - len(pp_cid_proposal) )/2)
            
    np.random.set_state(rng_state)
    return pp_cid_accepted.astype(int), pp_xy_accepted.astype(int)

def best_fitting_grid(n, ratio):
    '''
    Given a number of points n, find the rectangle containing regularily spaced
    points whose ratio of side lengths is closest to the "ratio" variable

    Parameters
    ----------
    n : int
        Number of points.
    ratio : float
        Desired ratio between side lengths of rectangle.

    Returns
    -------
    best_grid : tuple
        Number of rows and columns for "best" rectangle.

    '''
    min_diff = float('inf')
    best_grid = (0, 0)

    for R in range(1, int(n/2)):
        for multiplier in [-1, 0, 1]:
            C = ratio * R + multiplier
            if C > 0:
                total_points = R * C
                diff = abs(total_points - n)
                if diff < min_diff or (diff == min_diff and multiplier == 0):
                    min_diff = diff
                    best_grid = (R, C)
    
    return best_grid

def create_pilot_points_even(gwf, pars:dict,):
    '''
    Creates regularily spaced pilot points

    Parameters
    ----------
    gwf : flopy object
        Groundwater flow object (Modflow 6 model).
    pars : dict
        Simulation / Model parameters.

    Returns
    -------
    arrays of float
        Cell ID's and xy coordinates of pilot points.

    '''
    rng_state = np.random.get_state()
    np.random.seed(805)
    nPP = pars['n_PP']
    mg = gwf.modelgrid
    ixs  = flopy.utils.GridIntersect(mg, method = "vertex")
    extent = mg.extent
    xyz = mg.xyzcellcenters
    xy = list(zip(xyz[0], xyz[1]))
    
    welxy = pars['welxy']
    obsxy = pars['obsxy']
    
    welcid = ixs.intersect(MultiPoint(welxy))
    obscid = ixs.intersect(MultiPoint(obsxy))
    
    blocked_cid = np.concatenate((welcid.cellids,obscid.cellids))
    pp_cid_accepted = []
    pp_xy_accepted = []
    
    ratio = int(extent[1] / extent[3])
    
    nPPy, nPPx = best_fitting_grid(nPP, ratio)
    nPPprod = nPPy * nPPx
    
    xext = 1/(nPPx*2)
    yext = 1/(nPPy*2)
    xratio = np.linspace(xext, 1-xext, nPPx)
    yratio = np.linspace(yext, 1-yext, nPPy)
    offsetx, offsety  = 0, 0
    count = 1
    
    while len(pp_cid_accepted) != nPPprod:
        PPxloc = extent[1] * xratio + offsetx
        PPyloc = extent[3] * yratio + offsety
        
        pp_xy_proposal = [(x, y) for x in PPxloc for y in PPyloc]
        
        pp_cid_proposal = np.zeros(len(pp_xy_proposal))
        for i, point in  enumerate(pp_xy_proposal):
            pp_cid_proposal[i] = ixs.intersect(Point(point)).cellids.astype(int)[0]
        
        common_cells = np.intersect1d(pp_cid_proposal, blocked_cid)
        pp_cid_accepted = np.setdiff1d(pp_cid_proposal, common_cells)
        
        if len(pp_cid_accepted) < nPPprod:
            if count < 10:
                if count%2 == 0:
                    offsety = np.random.randn() * pars['dx'][1]
                    offsetx = 0
                else:
                    offsetx = np.random.randn() * pars['dx'][0]
                    offsety = 0
            else:
                offsety = np.random.randn() * pars['dx'][1]
                offsetx = np.random.randn() * pars['dx'][0]
            count += 1
        if len(pp_cid_accepted) == nPPprod:
            pp_xy_accepted = np.array([xy[int(i)] for i in pp_cid_accepted])

    np.random.set_state(rng_state)
    return pp_cid_accepted.astype(int), pp_xy_accepted.astype(int)








