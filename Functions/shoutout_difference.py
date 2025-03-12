import numpy as np
def shout_dif(array1, array2):
    
    
    meandiff = np.mean(np.abs(array1-array2))
    maxdiff = np.max(np.abs(array1-array2))
    mindiff = np.min(np.abs(array1-array2))
    
    print(f"Average difference: {meandiff:.4f}")
    print(f"Maximum difference: {maxdiff:.4f}")
    print(f"Minimum difference: {mindiff:.4f}")
    