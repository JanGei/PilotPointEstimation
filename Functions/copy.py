#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains functions that create directories for the ensemble 
members, as well as multiply the original model.

@author: janek geiger
"""

import os
import shutil

def create_Ensemble(pars: dict, iso = False) -> list:
    '''
    Creates ensemble by copying the reference model to the ensemble directory

    Parameters
    ----------
    pars : dict
        Simulation / model parameters.
    iso : bool, optional
        Flag whether ensemble is isotropic or anisotropic. The default is False.

    Returns
    -------
    list
        Conatins model directories for all ensemble members.

    '''
    n_mem       = pars['n_mem']
    ens_m_dir = []
    
    # removing old Ensemble
    if iso:
        e_ws = pars['isoens']
        m_ws = pars['isomem']
    else:
        e_ws = pars['ens_ws']
        m_ws = pars['mem_ws']
        
    if os.path.exists(e_ws) and os.path.isdir(e_ws):
        shutil.rmtree(e_ws)
        os.mkdir(e_ws)
    else:
        os.mkdir(e_ws)
    
    for i in range(n_mem):
        mem_dir = m_ws + f'{i}'
        # Copy the steady_state model folder to new folders
        shutil.copytree(pars['sim_ws'], mem_dir)
        ens_m_dir.append(mem_dir)

    return ens_m_dir



def copy_model(orig_dir:str, model_dir: str) -> None:
    '''
    Copies a model from one to another directory.

    Parameters
    ----------
    orig_dir : str
        Path of model to copied.
    model_dir : str
        Destination path.

    Returns
    -------
    None

    '''
    # Check if the destination folder already exists
    if os.path.exists(model_dir):
        # Remove the existing destination folder and its contents
        shutil.rmtree(model_dir)

    # Copy the model folder to new folder
    shutil.copytree(orig_dir, model_dir)
