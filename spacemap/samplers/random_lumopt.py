import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from lumopt.optimization import Optimization
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.optimizers.generic_optimizers import ScipyOptimizers

class RandomLumopt:
    """
    Define a sampler based on lumopt inverse design with random restart
        
    Parameters
    ----------
    settings: Parameters
        List of settings to initialize the study
    study: obj
        The study to use
    """
    
    def __init__(self, settings,study):
        """ Settings are stored, default values added if needed """
        
        # Store the study #
        ###################
       
        self._study = study
        self._parameters_size = self._study.geometry.parameters_size
        
        # Read settings #
        #################            
        if hasattr(settings, 'global_sample_function'):
            # Use given function and ignore bounds
            self._global_sample_function = settings.global_sample_function
            self._global_parameters_bounds = None
        else:
            # If no function, use uniform rand with given boundaries if provided. If not, assume [0,1]
            if hasattr(settings, 'global_parameters_bounds'):
                self._global_parameters_bounds = np.array(settings.global_parameters_bounds)
            else:
                self._global_parameters_bounds = [(0, 1)]*self._parameters_size
            
            self._global_sample_function = lambda: self._global_parameters_bounds[:,0] + (self._global_parameters_bounds[:,1]-self._global_parameters_bounds[:,0])*np.random.rand(1,self._parameters_size).flatten()
            

        if hasattr(settings, 'global_result_constraint'):
            self._global_result_constraint = settings.global_result_constraint
        else:
            self._global_result_constraint = None           
       
        if hasattr(settings, 'local_result_constraint'):
            self._local_result_constraint = settings.local_result_constraint
        else:
            self._local_result_constraint = None
            
        if hasattr(settings, 'local_max_iterations'):
            self._local_max_iterations = settings.local_max_iterations
        else:
            self._local_max_iterations = 50
            
        if hasattr(settings, 'local_method'):
            self._local_method = settings.local_method
        else:
            self._local_method = 'L-BFGS-B'
        
        if hasattr(settings, 'local_scaling_factor'):
            self._local_scaling_factor = settings.local_scaling_factor
        else:
            self._local_scaling_factor = 1
            
        if hasattr(settings, 'local_ftol'):
            self._local_ftol = settings.local_ftol
        else:
            self._local_ftol = 1e-5
            
        if hasattr(settings, 'local_pgtol'):
            self._local_pgtol = settings.local_pgtol
        else:
            self._local_pgtol = 1e-5
        
        # Wavelength settings for lumopt           
        if hasattr(settings, 'local_wavelength_start'):
            self._local_wavelength_start = settings.local_wavelength_start
        else:
            self._local_wavelength_start = 1550e-9
        
        if hasattr(settings, 'local_wavelength_stop'):
            self._local_wavelength_stop = settings.local_wavelength_stop
        else:
            self._local_wavelength_stop = 1550e-9
            
        if hasattr(settings, 'local_wavelength_points'):
            self._local_wavelength_points = settings.local_wavelength_points
        else:
            self._local_wavelength_points = 1
        
        # How many times the sampler runs
        self._iteration = 0
        self._results_num = 0
        self._sampling_param = list()
        self._sampling_res = list()
            
            
    def run(self, max_results, reset_counter=False, max_iter=np.Inf):
        """
        Run the sampling
        
        Parameters
        ----------
        max_results: int
            Number of required optimized results (eventually respecting local_result_constraint)
        reset_counter: bool
            If true, reset the counters of the number of iterations and results to zero and clears the results previously obtained
        max_iter: int
            Maximum number of samples

        Returns
        ----------
        results: list
            The fom from all the optimizer runs
        parameters: list
            The corresponding parameter sets
        """
        
        
        if reset_counter:
            self._iteration = 0
            self._results_num = 0
            self._sampling_param = list()
            self._sampling_res = list()
            
        # Run a first simulation with initial parameters already stored in the geometry
        new_param = None
        
        # keep track of consecutive errors
        error_flag = 0
            
        
        while self._iteration < max_iter and self._results_num < max_results and error_flag < 6:
            try:
                print("Global search - iteration " + str(self._iteration))
                print("Acceptable results: " + str(self._results_num))
            
                if self._global_result_constraint is not None:
                    # Only run simulation if we are screening starting points in the global stage
                    sim_res, sim_param = self._study.run(param=new_param)
                    self._sampling_param.append(sim_param)
                    self._sampling_res.append(sim_res)
                else:
                    # Otherwise only update the geometry
                    self._study.update_geometry(param=new_param)
                
                if (self._global_result_constraint == None) or (self._global_result_constraint(sim_res)):
                
                    current_folder = os.getcwd() + '\\'
                
                    # Inverse design, create. We have to do this here because in self.samples.geometry.geometry
                    # the parameters are already updated to the last global search.
                    inverse_design = LumericalInverseDesign(max_iter = self._local_max_iterations, 
                                                            method = self._local_method,
                                                            scaling_factor = self._local_scaling_factor,
                                                            pgtol = self._local_pgtol,
                                                            ftol = self._local_ftol,
                                                            wavelength_start = self._local_wavelength_start,
                                                            wavelength_stop = self._local_wavelength_stop,
                                                            wavelength_points = self._local_wavelength_points,
                                                            build_simulation = self._study.simulation_builder,
                                                            fom = self._study.fom.fom,
                                                            geometry = self._study.geometry.geometry,
                                                            hide_fdtd_cad = self._study.sim.hide_gui)
                    
                    # Inverse design, run
                    res = inverse_design.run()
                    
                    # Store optimization result
                    self._sampling_param.append(np.array(res[1]))
                    self._sampling_res.append(res[0])
                
                    # Return to proper folder
                    os.chdir(current_folder)
                    
                    # Inverse design, clear
                    del inverse_design
                
                    # Only counts as results if constraint are satisfied
                    if (self._local_result_constraint == None) or (self._local_result_constraint(res[0])):
                        self._results_num += 1
                
                # Random restart with constraint check: update new_param
                flag_param_constraint = False
                while not flag_param_constraint:
                    new_param = self._global_sample_function()
                    
                    if np.array(self._global_parameters_bounds == None).all() :
                        flag_param_constraint = True
                    else:
                        if (all(new_param>self._global_parameters_bounds[:,0]) and all(new_param<self._global_parameters_bounds[:,1])):
                            flag_param_constraint = True
            
                self._iteration += 1
                
                # simulation completed, reset error flag
                error_flag = 0
                
            except Exception as e:
                print("Error random_lumopt module, run function: ", end="")
                print(e)
                error_flag += 1
            
        
        # Sampling is done, close the simulation interface
        self._study.close_simulation()
        
        return(self._sampling_res, self._sampling_param, self._results_num)
            
            
            


class LumericalInverseDesign:
    """
    Wrapper for Lumerical lumopt (part of it)
        
    Parameters
    ----------
    max_iter: int
    method: string
    scaling_factor: float
    pgtol: float
    ftol: float
    wavelength_start: float
    wavelength_stop: float
    wavelength_points: float
    build_simulation: string
    fom: obj
    geometry: obj
    hide_fdtd_cad: bool

    """
    
    def __init__(self, max_iter, method, scaling_factor, pgtol, ftol, wavelength_start, wavelength_stop,
                 wavelength_points, build_simulation, fom, geometry, hide_fdtd_cad):
        
        # The optimizer must be generated anew at each iteration
        self._new_local_optimizer = ScipyOptimizers(max_iter=max_iter, method=method,
                                                   scaling_factor=scaling_factor, ftol=ftol, pgtol=pgtol)
        
        self._wl = Wavelengths(start = wavelength_start, stop = wavelength_stop, points = wavelength_points)
        
        self._optimization = Optimization(base_script = build_simulation,
                                          wavelengths = self._wl,
                                          fom = fom,
                                          geometry = geometry,
                                          optimizer = self._new_local_optimizer,
                                          hide_fdtd_cad = hide_fdtd_cad,
                                          use_deps = True)
        
    def run(self):
        """
        Run the lumopt optimization
        
        Returns
        -------
        res: array
            The figure of merit of the optimized device
        param: numpy array
            The optimized parameters

        """
        results = self._optimization.run()
        self._optimization.sim.fdtd.close()
        
        # plot optimization recap figure
        plt.show()
        
        return [results[0], np.array(results[1])]
    
    
    def _cleanup(self):
        ''' Remove all the folders generated by lumopt '''
        
        # folder for the file
        local_folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        subdir_list = os.listdir(local_folder)
        
        for folder in subdir_list:
            if folder.startswith('opts_') or (folder.startswith('optimization') and folder.endswith('.png')):
                shutil.rmtree(local_folder+'\\'+folder, ignore_errors=True)
                
    
    def __del__(self):
        # Remove  objects to delete pointers or pickle could have problems
        del self._optimization
        del self._wl
        del self._new_local_optimizer
        
        self._cleanup()
        
        