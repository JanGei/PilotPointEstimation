# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:01:12 2023

@author: Janek
"""
import flopy
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.utils.gridgen import Gridgen
from shapely.geometry import LineString, MultiPoint
import numpy as np
import shutil
from Field_Generation import generate_fields
from random_smooth_step import smooth_step
from convert_transient import convert_to_transient
from plot_fields import plot_fields
from model_params import get
import sys
import os
from itertools import repeat

def create_reference_model(pars):
# if __name__ == '__main__':
    pars = get()
    #%% Model Parameters
    nx      = pars['nx']
    dx      = pars['dx']
    toph    = pars['top']
    nlay    = pars['nlay'][0]
    mname   = pars['mname']
    sname   = pars['sname']
    sim_ws  = pars['sim_ws']
    gg_ws   = pars['gg_ws']
    
    #%% Grid Generation
    Lx = nx[0] * dx[0]
    Ly = nx[1] * dx[1]
    
    delr = np.ones(nx[0])*Lx/nx[0]
    delc = np.ones(nx[1])*Ly/nx[1]
    
    top     =  np.array([np.ones((nx[1],nx[0]))]*toph)
    botm    =  np.array([np.zeros((nx[1],nx[0]))])
    
    strgrd = StructuredGrid(delc=delc.astype(int), delr=delr.astype(int), top=top, botm=botm, nlay=nlay)
    if os.path.exists(gg_ws):
        shutil.rmtree(gg_ws)
    g = Gridgen(strgrd, model_ws=gg_ws)
    #%% Well Location
    welxy   = pars['welxy']
    welq    = pars['welq']
    welay   = pars['welay']
    
    if pars['refine']:
        # possible refinements
        g.add_refinement_features(welxy, "point", 4, range(nlay))
    else:
        os.mkdir(gg_ws)
    
    #%%  Boundary - river
    river           = pars['river']
    rivd            = pars['rivd']
    river_stages    = np.genfromtxt(pars['rh_d'],delimiter = ',', names=True)['Wert']
    rivC            = pars['rivC']
    riv_line        = [tuple(xy) for xy in river]
    riv_gradient    = pars['rivgrd']
    
    if pars['refine']:
        # possible refinements
        g.add_refinement_features([riv_line], "line", 3, range(nlay))
    
    #%% Northern Boudnary - Fixed head
    chdls            = pars['chd']
    chd_stages       = pars['chdh']
    # chd_line       = [tuple(xy) for xy in river]
    
    # %% Southern Boundary Drainage
    drnls = LineString(pars['drn'][0])
    drn_stages = pars['drnh']
    drn_cond = pars['drnc']
    #%% Buildng Grid

    g.build()
    disv_props  = g.get_gridprops_vertexgrid()
    vgrid       = flopy.discretization.VertexGrid(**disv_props)
    idom        = np.ones([vgrid.nlay, vgrid.ncpl])
    strt        = np.zeros([vgrid.nlay, vgrid.ncpl])+20
    ixs         = flopy.utils.GridIntersect(vgrid, method = "vertex")
    
     
    #%% Flopy Model definiiton - Core packages
    
    # simulation object
    sim     = flopy.mf6.MFSimulation(sim_name           = sname,
                                     sim_ws             = sim_ws,
                                     verbosity_level    = 0)
    # groundwater flow / model object
    gwf     = flopy.mf6.ModflowGwf(sim,
                                   modelname            = mname,
                                   save_flows           = True)
    # disv package
    disv    = flopy.mf6.ModflowGwfdisv(model            = gwf,
                                       length_units     = "METERS",
                                       pname            = "disv",
                                       xorigin          = 0,
                                       yorigin          = 0,
                                       angrot           = 0,
                                       nogrb            = False,
                                       nlay             = disv_props["nlay"], 
                                       ncpl             = disv_props["ncpl"],
                                       nvert            = len(disv_props["vertices"]), 
                                       top              = disv_props["top"],
                                       botm             = disv_props["botm"], 
                                       idomain          = idom, 
                                       cell2d           = disv_props["cell2d"], 
                                       vertices         = disv_props["vertices"])
    
    # tdis package
    tdis    = flopy.mf6.ModflowTdis(sim,
                                    time_units          = "SECONDS",
                                    perioddata          = [[60*60*6, 1, 1.0]])
    
    # ims package
    ims = flopy.mf6.ModflowIms(sim,
                               print_option             = "SUMMARY",
                               complexity               = "COMPLEX",
                               linear_acceleration      = "BICGSTAB")
    
    # oc package
    headfile            = "{}.hds".format(mname)
    head_filerecord     = [headfile]
    budgetfile          = "{}.cbb".format(mname)
    budget_filerecord   = [budgetfile]
    saverecord          = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    printrecord         = [("HEAD", "LAST")]
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                saverecord              = saverecord,
                                head_filerecord         = head_filerecord,
                                budget_filerecord       = budget_filerecord,
                                printrecord             = printrecord)
    
    sto = flopy.mf6.ModflowGwfsto(gwf, 
                                  pname                 = "sto",
                                  save_flows            = True,
                                  iconvert              = 1,
                                  ss                    = pars['ss'],
                                  sy                    = pars['sy'],
                                  steady_state          = {0: True},
                                  transient             = {0: False})

    sim.write_simulation()
    #%% Generating and loading reference fields
    generate_fields(pars)
    print('Fields are generated')
    k_ref = np.loadtxt(pars['k_r_d'], delimiter = ',')
    r_ref = np.loadtxt(pars['r_r_d'], delimiter = ',')

    
    #%% Intersecting model grid with model features
    
    rch_cells       = np.arange(vgrid.ncpl)
    rch_lay         = np.zeros(vgrid.ncpl, dtype = int)
    rch_cell2d      = list(zip(rch_lay,rch_cells))
    
    if pars['rch_is']:
        rch_list        = list(zip(rch_cell2d, repeat(r_ref)))
    else:
        rch_list        = list(zip(rch_cell2d, abs(r_ref.flatten())))
        
    for i in range(vgrid.ncpl):
        rch_list[i] = list(rch_list[i])
        
    ### Wells
    result      = ixs.intersect(MultiPoint(welxy))
    well_list   = []
    for i, index in zip(result.cellids, range(len(result.cellids))):
        pump = welq[index].astype(float) #* (pars['welnd'][index]-pars['welst'][index])/360
        layer   = welay[index].astype(int)
        well_list.append([(layer,i),-pump])
        # set conductivity in well cells to 1
        if pars['wel_k']:
            k_ref[i] = 1
    
    
    
    #%% River
    # riverLS     = LineString(river)
    # l           = riverLS.length
    riv_list    = []
    result = ixs.intersect(LineString(river)).cellids
    result = np.array([int(r) for r in result], dtype=int)
    rivcellxy = np.array([vgrid.xyzcellcenters[0][result], vgrid.xyzcellcenters[1][result]])
    # sort in ascending x
    sorted_indices = np.argsort(rivcellxy[0]) 
    rivcellxy = rivcellxy[:, sorted_indices] 
    sorted_ids = result[sorted_indices]
    Conductance = smooth_step(rivcellxy[0,:], np.array([1200, 4900, 7200, 8400]), np.array([1.4, -1.1, 0.5, 1.8])*1e-3, alpha=0.01)+rivC*5
    dxriv = np.diff(rivcellxy[0,:])
    dyriv = np.diff(rivcellxy[1,:])
    dsriv = np.sqrt(dxriv**2 + dyriv**2)
    dsriv = np.insert(np.cumsum(dsriv), 0, 0)
    # Norm to original river length
    dsriv = dsriv / np.max(dsriv) * LineString(river).length
    
    for i, cell in enumerate(sorted_ids):
        stage = river_stages[0]+16-dsriv[i]*riv_gradient
        riv_list.append([(0, cell), stage, Conductance[i] , stage-rivd])
    
    #%% Drn package
    drn_list    = []

    result  = ixs.intersect(drnls)
    for cell in result.cellids:
        drn_list.append([(0, cell), drn_stages[0], drn_cond[0]])
    
            
    #%% Chd
    # chdLS       = LineString(chdl)
    # lchd        = chdLS.length
    chd_list    = []
    
    for i, chdl in enumerate(chdls):
        for ii in range(len(chdl)-1):
            chdls   = LineString(np.array([chdl[ii],chdl[ii+1]]))
            result  = ixs.intersect(chdls)
            for cell in result.cellids:
                # xc,yc = vgrid.xyzcellcenters[0][cell],vgrid.xyzcellcenters[1][cell]
                chd_list.append([(0, cell), chd_stages[i]])
    
    # npf package
    npf     = flopy.mf6.ModflowGwfnpf(model             = gwf,
                                      k                 = k_ref)

    # drn package
    ic = flopy.mf6.ModflowGwfic(gwf, 
                                strt                    = strt)
    # ic package
    drn = flopy.mf6.ModflowGwfdrn(gwf, 
                                  stress_period_data    = {0:drn_list})
    
    # rch package
    rch = flopy.mf6.ModflowGwfrch(gwf,
                                  stress_period_data    = {0:rch_list})
    # wel package
    wel = flopy.mf6.ModflowGwfwel(gwf,
                                  stress_period_data    = {0:well_list})
    # riv package
    riv = flopy.mf6.ModflowGwfriv(gwf,
                                  stress_period_data    = {0:riv_list})
    
    # chd package
    chd = flopy.mf6.ModflowGwfchd(gwf,
                                  stress_period_data    = {0:chd_list})
    
    
    #%% Set steady-state solution as initial condition
    sim.write_simulation()
    sim.run_simulation()
    ic.strt.set_data(gwf.output.head().get_data())
    ic.write()
    
    if pars['inspec'] and pars['setup'] == 'office':
        print(pars['mu'][0], np.mean(np.log(k_ref)))
        print(pars['sigma'][0], np.var(np.log(k_ref)))
        print(pars['mu'][1], np.mean(np.abs(r_ref*86400*1000)))
        print(pars['sigma'][1], np.var(np.abs(r_ref*86400*1000)))
        plot_fields(gwf, pars, np.log(k_ref), r_ref, POI = pars['obsxy'])
        print(f'Max K: {np.max(np.log(k_ref[k_ref <1]))}')
        print(f'Max K: {np.min(np.log(k_ref))}')
        print(f'Max R: {np.max(r_ref)}')
        print(f'Max R: {np.min(r_ref)}')
        gwf.ic.plot()
        sys.exit()
    
    #%% Run transient simulation
    convert_to_transient(sim_ws, pars['trs_ws'], pars)

    # transient_run(pars)
    
    # plot(gwf, ['logK', 'rch'])
    # plot(gwf, ['logK', 'rch', 'h'], bc = True)
    # plot(gwf, ['logK','h'], bc=False)

