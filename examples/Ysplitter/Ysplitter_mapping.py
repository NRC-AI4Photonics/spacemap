# Imports
# -----------------------------------------------------------------------------
import sys
# Spacemap location
sys.path.insert(0, "..\\..")
# lumopt location
sys.path.insert(0,"C:\\Program Files\\Lumerical\\2020a\\api\\python")

import spacemap as sm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch


# Service functions
# -----------------------------------------------------------------------------
# Function to generate the geometry. This function only takes parameters to be
# optimized as input
def taper_splitter(params):  
    
    delta = 0
    
    Nx = len(params)
    dx = 2e-6/Nx
    points_x = np.concatenate(([-1e-6], np.linspace(-1e-6+0.5*dx,1e-6-0.5*dx,Nx), [1e-6]))
    points_y = np.concatenate(([0.225e-6], params, [0.575e-6]))
    
    px = np.linspace(min(points_x), max(points_x), 100)
    interpolator = sp.interpolate.interp1d(points_x, points_y)
    py = interpolator(px)
    
    # Original spline
    interpolator = sp.interpolate.CubicSpline(points_x, points_y, bc_type = 'clamped')
    interpolator_prime = interpolator.derivative(nu=1)
    py = interpolator(px)
    pyp = interpolator_prime(px)
    
    theta = np.arctan(pyp)
    theta[0] = 0.
    theta[-1] = 0.
    
    px2 = px-delta*np.sin(theta)
    py2 = py+delta*np.cos(theta)
    px2[px2<px[0]] = px[0]
    px2[px2>px[-1]] = px[-1]
    
    polygon_points_up = [(x, y) for x, y in zip(px2, py2)]
    polygon_points_down = [(x, -y) for x, y in zip(px2, py2)]
    polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
    return polygon_points

param_bounds = [(0.1e-6, 1e-6)]*10


# Settings
# -----------------------------------------------------------------------------
parSet = sm.Settings()

## General settings
# A filename suffix
parSet.general.suffix = 'Ysplitter_data' 
# Use comments to keep track of simulator settings.
parSet.general.comments = 'Y splitter with 0.25 um output separation and spline interpolation'
# Autosave after each simulation
parSet.general.autosave = True

## Study settings
# Select study type
parSet.study.type = 'LumericalFDTD'
# Base file to setup initial simulation environment (lsf, fsp or python function)
parSet.study.simulation_builder = 'splitter_base_TE.lsf'
# Function to build the geometry to optimized
parSet.study.geometry_function = FunctionDefinedPolygon(func = taper_splitter,
                                                        initial_params = np.ones(10,)*0.75e-6,
                                                        bounds = param_bounds, 
                                                        z = 0,
                                                        depth = 220e-9,
                                                        eps_out = 1.44**2,
                                                        eps_in = 2.85**2,
                                                        edge_precision = 5,
                                                        dx = 0.1e-9)
# A name to identify the simulation results
parSet.study.fom_name = 'mode_match'
# Figure of merit
parSet.study.fom_function = ModeMatch(monitor_name = 'fom', mode_number = 2, direction = 'Forward')
# Hide GUI during simulation
parSet.study.hide_gui = False


## Sampler settings
parSet.sampler.type = 'random-lumopt'
# Parameters bounds for global search
parSet.sampler.global_parameters_bounds = param_bounds

# Function to filter simulation results after local search (optional)
parSet.sampler.local_result_constraint = lambda res: res > 0.95
# lumopt parameters
parSet.sampler.local_max_iterations = 50
parSet.sampler.local_ftol = 1e-3
parSet.sampler.local_pgtol = 1e-3
parSet.sampler.local_scaling_factor = 1e6
parSet.sampler.local_wavelength_start = 1530e-9
parSet.sampler.local_wavelength_stop = 1650e-9
parSet.sampler.local_wavelength_points = 11

## Dimensionality reduction settings
# DM algorithm to use (only PCA at the moment)
parSet.dimensionality_reduction.type = 'pca'
# The number of dimensions for the reduced space
parSet.dimensionality_reduction.n_components  = 3


# Create the study
# -----------------------------------------------------------------------------
mapping = sm.SpaceMapping(settings = parSet)

# Run sampling (global+local)
# -----------------------------------------------------------------------------
# Parameter: maximum number of local search run
sampling_done = mapping.run_sampling(max_results=10)


if sampling_done:
    
    # Dimensionality reduction
    # -------------------------------------------------------------------------
    # Define lower bound to filter search simulation results
    mapping.dimensionality_reduction(lower_bound = 0.95)


    # Mapping
    # -------------------------------------------------------------------------
    idx_map =  mapping.subspace_sweep(distance = 1e-6, go=True)
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
    print("Number of designs with efficiency larger than 0.95: ", end="")
    print(str(np.size(np.where(np.array(mapping.maps[idx_map].result)[:,0]>0.95))))
    print("Best efficiency: {:1.3f}".format(best_eff), end="")
    print(" [simulation index " + str(idx_best) + "]")

    # Plotting
    # -------------------------------------------------------------------------
    axis = np.array(mapping.maps[idx_map].projected_grid)
    fom = np.array(mapping.maps[idx_map].result)
    
    # Plot figure of merit in 3D subspace
    fig_1=plt.figure()
    ax = fig_1.add_subplot(111, projection='3d')
    points = ax.scatter(axis[:,0]*1e6,axis[:,1]*1e6,axis[:,2]*1e6, c=np.array(fom[:,0]), alpha=0.8, s=-1/np.log10(np.array(fom)), cmap='jet')
    cb = fig_1.colorbar(points)
    cb.set_label('Y branch efficiency')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.view_init(elev=30., azim=60)

    training_data = np.array(mapping.maps[idx_map].projected_training_parameters)
    ax.scatter(training_data[:,0]*1e6,training_data[:,1]*1e6,training_data[:,2]*1e6, c='k', s=30, marker='s')

    #Plot Y junction profiles
    plt.figure()
    plt.title('Y branch profiles (efficiency > 0.95)')
    plt.xlabel('X [um]')
    plt.ylabel('Y [um]')
    for idx, val in enumerate(fom):
        if val > 0.95:
            mapped_design = mapping.maps[idx_map].grid[idx]
            mapped_design = taper_splitter(mapped_design)
            plt.plot(mapped_design[:,0]*1e6, mapped_design[:,1]*1e6,linewidth=0.5)


    mapped_design = mapping.maps[idx_map].grid[idx_best]
    mapped_design = taper_splitter(mapped_design)        
    plt.plot(mapped_design[:,0]*1e6, mapped_design[:,1]*1e6,color='k', label = "Best efficiency {:1.3f}".format(best_eff))
    plt.legend()
