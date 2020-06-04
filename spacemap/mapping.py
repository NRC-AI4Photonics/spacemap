import os
import numpy as np
import dill

from .utilities.parameters import Parameters, Settings
from .utilities.data_collection import DataCollection
from .utilities.map import Map

from .studies.study import Study
from .samplers.sampler import Sampler
from .dimensionality.dimensionality_reduction import DimensionalityReduction



class SpaceMapping:
    """
    Class that combines all the pieces for space mapping.
        
    Parameters
    ----------
    settings: Parameters, None
        List of settings to initialize all required objects
        
    Attributes
    ----------
    set: Parameters
        General settings
    data: obj
        Data collection object (database) to store all simulation results
    study: obj
        The study object, define geometry, figure of merit, and simulator
    sampler: obj
        The object that samples the design space
    dr: obj
        The dimensionality reduction object
    maps: list
        List of 'Map' objects, each store the result of one mapping run
    """
    
    def __init__(self, settings = None):
        
        if settings == None:
            settings = Settings()            
        
        # Process general settings #
        ############################ 
        self.set = Parameters()
        self.set.suffix = settings.general.suffix      
        self.set.comments = settings.general.comments
        self.set.autosave = settings.general.autosave

        # Create the simulations database #
        ###################################
        self.data = DataCollection()
        
        
        # Create the study: geometry, FOM, and simulator #
        ##################################################
        self.study = Study.get_study(settings=settings.study, file_name=self.set.suffix)
        
        # Create the sampler #
        ######################
        self.sampler = Sampler.get_sampler(settings.sampler, study=self.study)
     
        # Create the dimensionality reduction #
        #######################################
        self.dr = DimensionalityReduction.get_dm(settings.dimensionality_reduction)
        
        # Mapping #
        ###########
        self.maps = list()
        
        # A container for private service variables #
        #############################################
        self._service = Parameters()
        self._service.sampler_iteration = 0
        self._service.sampler_results_num = 0
        self._service.dr_training_index = list()
        self._service.new_name_generated = False #used with autosave to avoid override with first simulation
        self._service.current_file_name = self.set.suffix
        self._service.current_folder = os.getcwd() + '\\'
        
        
        
    
    # setter and getter for self.maps variable #
    ############################################
    
    @property
    def maps (self):
        returning_maps = list()
        for cur_map in self._maps:
            m = Parameters()
            m.model = cur_map.model
            m.training_parameters_idx = cur_map.training_data_idx
            
            # It is identical to cur_map.model.training_data
            m.training_parameters = [self.data.sample[i].parameters for i in cur_map.training_data_idx] 
            m.projected_training_parameters = cur_map.model.projected_training_data
            
            m.grid = [self.data.sample[i].parameters for i in cur_map.sim_idx]
            m.projected_grid = cur_map.projected_grid
            m.normalized_projected_grid = cur_map.normalized_projected_grid
            m.result = [self.data.sample[i].result for i in cur_map.sim_idx]
            
            returning_maps.append(m)
  
        return returning_maps
        
    @maps.setter
    def maps(self,new_map):
        if type(new_map)==list:
            self._maps = new_map
        else:
            self._maps.append(new_map)    
    
    
    # Run the study once and save the project file if requested #
    #############################################################

    def run_study(self, param=None, name = 'default', force_run = False):
        """ 
        Execute the study with the given parameters
        
        Paramenters
        -----------
        param: list, None
            The parameter set for the simulation
        name: str
            A name to identify the simulation. 'default', 'sweeping', and 'sampling'
            are reserved for use in various methods in this class
        force_run: bool
            Execute the simulation even in the case the same set of parameters has already been simulated
            
        Returns
        -------
        res: list
            The simulation result
        sample_idx: int
            The index of the simulation result in the database 'data'   
        
        """
        
        # Check if the combination parameters/fom has already been tested
        if (np.array(param == None)).any():
            already_simulated = False
            sample_idx = 0
        else:
            already_simulated, sample_idx = self.data.is_sample(param, self.study.fom_name)

        
        # Run the study
        if not already_simulated or force_run:
            
            res, used_param = self.study.run(param=param)
            
            # If simulation OK, store the new simulation in the database
            if (res != False):
                sample_idx = self.data.add_sample(parameters=used_param, simulation_name=name,
                                                  result=res, result_name=self.study.fom_name)    
            
        else:
            res = self.data.sample[sample_idx].result
        
        # Save
        if self.set.autosave and (res is not False):
            self.save_data(override = True)
            
        return (res, sample_idx)
    

    

    # Run the sampler and save if requested #
    #########################################
    
    def run_sampling(self, max_results=100, reset=False, max_iter=np.Inf):
        
        if reset:
            self._service.sampler_iteration = 0
            self._service.sampler_results_num = 0
            
        # Keep track of the number of errors
        error_num = 0
        
        # Run the sampling
        while (self._service.sampler_iteration < max_iter and 
               self._service.sampler_results_num < max_results and 
               error_num < 6):
            
            print("Global search - iteration " + str(self._service.sampler_iteration))
            print("Acceptable results: " + str(self._service.sampler_results_num))
            
            res, used_param, good_result, error_flag = self.sampler.run()
            
            # check if error
            if error_flag == True:
                error_num =+ 1
            else:
                
                # no error occured, reset the flag, increment the iteration counter,
                # and store the result(s) in the database
                
                error_num = 0
                self._service.sampler_iteration += 1
                
                for i in range(0,len(res)):

                    if (res != False):
                        self.data.add_sample(parameters=used_param[i], simulation_name='sampling',
                                             result=res[i], result_name=self.study.fom_name) 
                        
                # last, check if the optimized results is above threshold
                if good_result == True:
                    self._service.sampler_results_num = self._service.sampler_results_num + 1

            # Save
            if self.set.autosave:
                self.save_data(override = True)
        
        
        if self._service.sampler_results_num < max_results:
            # Something happend and sampler terminated earlier
            return False
        else:
            return True
             
                  
    # Run dimensionality reduction #
    ################################
    
    def dimensionality_reduction(self, result_name = None, result_index = 0,
                                 lower_bound = None, upper_bound = None):       
        
       
        # If result_name is None (no name specified) copy the current fom name
        if result_name == None:
            result_name = self.study.fom_name[:]
        
        # Filter the samples to use. Indices pointing to training data in the
        # database are stored. Only consider samples from 'sampling' stage, not 'sweeping'
        training_samples, _, self._service.dr_training_index = self.data.filter_simulation(result_name = result_name,
                                                                                     result_index = result_index,
                                                                                     simulation_name='sampling',
                                                                                     lower_bound = lower_bound,
                                                                                     upper_bound = upper_bound)
               
        # Train the model
        self.dr.train_model(training_data = training_samples)
        

    
    # Run a sweep of the reduced space and save if requested #
    ##########################################################
    
    def subspace_sweep(self, distance = None, n_points = None, boundaries = None, grid = None, go = False):
        
        # If grid is not provided, generate one
        if grid is None:
            grid, projected_grid, normalized_projected_grid = self.dr.subspace_mesh(distance, n_points, boundaries)
        else:
            # unpack the tuple
            grid, projected_grid, normalized_projected_grid = grid
        
        if go == False:
            response = input("The map has {} points, continue? [y,n] ".format(len(grid)))
            if response!='y':
                return None
        
        try:        
            mapping_idx = list()
          
            for param_idx, param in enumerate(grid):
                print("Sweeping - " + str(len(grid)) + " iterations: " + str(param_idx))          
                res, sim_idx = self.run_study(param = param, name = 'sweeping')
                if res == False:
                    raise Exception("Simulation failed and returned False value.") 
                mapping_idx.append(sim_idx) #store the database indices of the simulation
            
                     
        
            # Store also the dimensionality reduction model and the indices of the training data
            map_obj = Map(model = self.dr,
                          training_data_idx = self._service.dr_training_index,
                          projected_grid = projected_grid,
                          normalized_projected_grid = normalized_projected_grid,
                          sim_idx = mapping_idx)
        
            #append is done in the setter
            self.maps = map_obj 
        
        
            if self.set.autosave:
                self.save_data(override = True)  
                
                return len(self.maps)-1
        except Exception as e:
            print(e)
            print("Subspace sweeping terminated.")
 
    
    # Service functions #
    #####################    
    def save_data(self, file_name = None, override = False, csv = False):
        ''' If csv == True it only exports the database in csv format'''
        if file_name == None:            
            if csv == True:
                file_name = self._service.current_file_name + '.csv'
            else:
                file_name = self._service.current_file_name + '.pkl'
            
            if os.path.isfile(self._service.current_folder+file_name) and (override == False or self._service.new_name_generated == False):
                n = 0
                while os.path.isfile(self._service.current_folder+file_name):
                    n += 1
                    if csv == True:
                        file_name = self.set.suffix + '_' + str(n) + '.csv'
                    else:
                        file_name = self.set.suffix + '_' + str(n) + '.pkl'
                
                self._service.current_file_name = self.set.suffix + '_' + str(n)
                
            self._service.new_name_generated = True #only the first time I save I eventually override the override flag with new_name_generated flag
            file_name = self._service.current_folder+file_name
            flag_write = True
        
        else:
            if override == False and os.path.isfile(file_name):
                response = input('File ' + file_name + ' exists. Continue [y/n]? ')
                
                if response == 'y':
                    flag_write = True
                else:
                    flag_write = False
            else:
                flag_write = True
      
        
        if flag_write and csv == False:
            with open(file_name, 'wb') as f:
                # Export general settings
                dill.dump(self.set,f)
                # Export database
                dill.dump(self.data,f)
                # Export dimensionality reduction
                dill.dump(self.dr,f)
                # Export service variables
                dill.dump(self._service,f)
                # Export maps (WARNING: have to export _maps NOT maps. The latter is generated on the fly)
                dill.dump(self._maps,f)
                try:
                    # Export the study
                    dill.dump(self.study,f)
                except:
                    print('Study not saved, any interface open?') 
                try:
                    # Export the sampler
                    dill.dump(self.sampler,f)
                except:
                    print('Sampler not saved, any interface open?') 
       
        elif flag_write and csv == True:
            self.data.export_to_csv(file_name)
                
    def load_data(self, file_name = None, override=False):
        if override == False:
            response = input('Warning, this will override all settings, data, and function calls in the current study. Continue [y/n]? ')
        else:
            response = 'y'
        if response == 'y':
            with open(file_name, 'rb') as f:
                self.set = dill.load(f)
                self.data = dill.load(f)
                self.dr = dill.load(f)
                self._service = dill.load(f)
                self._maps = dill.load(f)
                try:
                    self.study = dill.load(f)
                except:
                    print('Study not available')
                try:
                    self.sampler = dill.load(f)
                except:
                    print('Sampler not available')
            self.print_report()
        return self

    
    def append_from_csv(self, file_name, parameters_size):
        """ Append results from a compatible csv file in the current database """
        self.data.load_from_csv(file_name, parameters_size)

    def print_report(self):
        """ Print a report of current settings """
        print ("================================")
        print ("  SpaceMapping current settings")
        print ("================================\n")
        
        print('suffix ')
        print('-------\n ' + self.set.suffix + '\n')
        print('comments')
        print('--------\n' + self.set.comments + '\n')   
        print('autosave')
        print('--------\n' + str(self.set.autosave) + '\n') 
        print('available maps')
        print('--------------\n' + str(len(self.maps)) + '\n')
        

        

