import numpy as np

def chd_mask(gwf):
    
    chd = gwf.chd.stress_period_data.get_data()
    head = gwf.output.head().get_data().flatten()
    chd_cid = list(map(lambda tup: tup[1], chd[0]['cellid']))
    mask = np.zeros(head.shape)
    mask[chd_cid] = int(1)
    
    return mask