import os
import numpy as np
def write_file(pars, data, names, decimals, intf = False):
    for i, element in enumerate(data):
        file_path = os.path.join(pars['resdir'], str(names[i]) + '.dat')
        if os.path.exists(file_path):
            os.remove(file_path)
        
        with open(file_path, 'a') as f:
            if isinstance(element, np.ndarray):
                if len(element.shape) == 1:
                    for k in range(element.shape[0]):
                        entry = element[k]
                        if intf:
                            f.write(f"{entry.astype(int)} ")
                        else:
                            f.write(f"{entry:.{decimals}f} ")
                    f.write('\n')
                elif len(element.shape) == 2:
                    for j in range(element.shape[0]):
                        for k in range(element.shape[1]):
                            entry = element[j,k]
                            if intf:
                                f.write(f"{entry.astype(int)} ")
                            else:
                                f.write(f"{entry:.{decimals}f} ")
                        f.write('\n')
            else:
                for entry in element:
                    f.write(f"{entry:.{decimals}f} ")
