# Imports
# -----------------------------------------------------------------------------
import sys
# Spacemap module location
sys.path.insert(0, "..\\..")
# lumopt module location
sys.path.insert(0,"C:\\Program Files\\Lumerical\\2020a\\api\\python")

import spacemap as sm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch


# Service functions
# -----------------------------------------------------------------------------
        
# Parameters to build the geometry
n_grates = 20
wg_height = 220.0e-9
etch_depth_frac = 0.5
y0 = 0
x0 = 0

# Geometry definition
def grate_function(param):    
    y2 = y0 + wg_height
    y1 = y2 - etch_depth_frac * wg_height
    
    #accept parameters in um
    param = param*1e-6
    
    verts = np.array([[x0,y0]])
    
    xp = x0
    for idx in range(n_grates):
        new_verts = np.array([[xp+param[0],y0],
                              [xp+param[0],y2],
                              [xp+param[0]+param[1],y2],
                              [xp+param[0]+param[1],y0],
                              [xp+param[0]+param[1]+param[2],y0],
                              [xp+param[0]+param[1]+param[2],y1],
                              [xp+param[0]+param[1]+param[2]+param[3],y1],
                              [xp+param[0]+param[1]+param[2]+param[3],y2],
                              [xp+param[0]+param[1]+param[2]+param[3]+param[4],y2],
                              [xp+param[0]+param[1]+param[2]+param[3]+param[4],y0]])
        
        
        verts = np.concatenate((verts, new_verts), axis = 0)
        xp += np.sum(param[0:5])
        
    verts = np.concatenate((verts, [[x0+param[0], y0]]), axis = 0)
    verts = np.flip(verts,axis=0)
    return verts


grating_geometry = FunctionDefinedPolygon(func = grate_function,
                                          initial_params = np.array([0.077, 0.084, 0.115, 0.249, 0.171]),
                                          bounds = [(0.04, 0.4)] * 5,
                                          z = 0.0,
                                          depth = wg_height,
                                          eps_out = 1.45 ** 2,
                                          eps_in = 3.48 ** 2,
                                          edge_precision = 5,
                                          dx = 1.0e-5)

# Define settings
# -----------------------------------------------------------------------------
parSet = sm.Settings()


## General settings
# A filename suffix
parSet.general.suffix = 'grating_coupler' 
# Use comments to keep track of simulator settings.
parSet.general.comments = 'L-shaped vertical grating coupler design'
# Autosave simulation results
parSet.general.autosave = True


## Study settings
# Select study type
parSet.study.type = 'LumericalFDTD'
# Name study parameters (useful expecially when exporting data)
parSet.study.parameters_name = ['L1','L2','L3','L4','L5']
# Base file to setup initial simulation environment (lsf, fsp or python function)
parSet.study.simulation_builder = 'GratingCoupler_base_setup.fsp'
# Function to build the geometry for optimization
parSet.study.geometry_function = grating_geometry
# A name to identify the simulation results (fom can change later on)
parSet.study.fom_name = 'mode_match'
# Fom function
parSet.study.fom_function = ModeMatch(monitor_name = 'fiber_monitor', mode_number = 1,
                                      direction = 'Forward', multi_freq_src = False,
                                      target_T_fwd = lambda wl: 1 * np.ones(wl.size), norm_p = 1)
# Hide GUI during simulation
parSet.study.hide_gui = False
# Close simulation interface after each random restart
parSet.study.simulator_restart = False


## Sampler settings 
parSet.sampler.type = 'random-lumopt'
# Parameters bounds during global search
parSet.sampler.global_parameters_bounds = param_bounds = [(0.04, 0.4)]*5
# Function to filter simulation results in global search before starting lumopt
parSet.sampler.global_result_constraint = lambda res: res > 0.1
# Function to filter simulation results after a lumopt run
parSet.sampler.local_result_constraint = lambda res: res > 0.7
# lumopt settings
parSet.sampler.local_method = 'L-BFGS-B'
parSet.sampler.local_max_iterations = 50
parSet.sampler.local_ftol = 1e-4
parSet.sampler.local_pgtol = 1e-4
parSet.sampler.local_scaling_factor = 1
parSet.sampler.local_wavelength_start = 1545e-9
parSet.sampler.local_wavelength_stop = 1555e-9
parSet.sampler.local_wavelength_points = 3


## Dimensionality reduction parameters 
# DM algorithm to use (only PCA at the moment)
parSet.dimensionality_reduction.type = 'pca'
# The number of dimensions for the reduced space
parSet.dimensionality_reduction.n_components  = 2
# Scale the standard deviation of all study's parameters to one before running PCA
parSet.dimensionality_reduction.scale = False


# Create the study
# -----------------------------------------------------------------------------
mapping = sm.SpaceMapping(settings = parSet)

# Run sampling (global+local)
# -----------------------------------------------------------------------------
# Parameter: maximum number of locally optimized results (if defined, only results respecting
# 'local_result_constraint' are counted)
sampling_done = mapping.run_sampling(max_results=5)


if sampling_done:
    
    # Dimensionality reduction
    # -------------------------------------------------------------------------
    # Define lower bound to filter search simulation results
    mapping.dimensionality_reduction(lower_bound = 0.70)


    # Mapping
    # -------------------------------------------------------------------------
    idx_map =  mapping.subspace_sweep(distance = 0.1, go=True)
    print("\nMap index: " + str(idx_map))

    # Export all simulation to csv
    # -------------------------------------------------------------------------
    mapping.save_data(csv=True)

    # Report
    # -------------------------------------------------------------------------
    best_eff = np.max(mapping.maps[idx_map].result)
    idx_best = np.argmax(mapping.maps[idx_map].result)

    print("\nMapping report")
    print("==============")
    print("Training results used for dimensionality reduction: " + str(len(mapping.maps[idx_map].training_parameters_idx)))
    print("Number of mapped designs with coupling efficiency larger than 0.7: ", end="")
    print(str(np.size(np.where(np.array(mapping.maps[idx_map].result)[:,0]>0.7))))
    print("Best efficiency: {:1.3f}".format(best_eff), end="")
    print(" [simulation index " + str(idx_best) + "]")

    # Plotting
    # -------------------------------------------------------------------------
    axis = np.array(mapping.maps[idx_map].projected_grid)
    fom = np.array(mapping.maps[idx_map].result)
    
    # Plot figure of merit in 3D subspace
    fig_1 = plt.figure()
    ax = fig_1.add_subplot(111)
    points = ax.scatter(axis[:,0],axis[:,1], c=np.array(fom[:,0]), alpha=0.8, s=-1/np.log10(np.array(fom)), cmap='jet')
    cb = fig_1.colorbar(points)
    cb.set_label('Coupling efficiency')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')