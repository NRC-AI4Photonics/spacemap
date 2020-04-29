import numpy as np
import os
from .lumerical_geometry import LumericalGeometryObject
from .lumerical_fom import LumericalFomObject
from .lumerical_fdtd_simulation import LumericalFDTD

class LumericalFDTDStudy:
    """
    Define a study based on lumopt and lumapi
        
    Parameters
    ----------
    settings: Parameters
        List of settings to initialize the study
    file_name: str
        File name used to save the fsp file before the simulation
    
    Attributes
    ----------
    parameters_name: array, None
        1D array of strings with parameters names
    fom_name: string
        Label for the returned simulation result
    geometry: Geometry object
        Object to store the geometry to be optimized
    fom: Fom object
        Object to retrive simulation results
    sim: LumericalFDTD object
        Interface to Lumerical FDTD simulator
    simulation_builder: string, function
        How to setup the simulator
    """
    
    def __init__(self, settings, file_name='default'):
        """ Settings are stored, default values added if needed """
        
        current_folder = os.getcwd() + '\\'
        
        # Read settings #
        #################
    
        if hasattr(settings, 'initial_parameters'):
            initial_parameters = settings.initial_parameters
        else:
            initial_parameters = None
            
        if hasattr(settings, 'parameters_name'):
            self.parameters_name = settings.parameters_name
        else:
            self.parameters_name = None
            
        if hasattr(settings, 'geometry_function'):
            geometry_function = settings.geometry_function
        else:
            geometry_function = lambda x: True
        
        if hasattr(settings, 'fom_name'):
            self.fom_name = settings.fom_name
        else:
            self.fom_name = 'fom'
        
        if hasattr(settings, 'fom_function'):
            fom_function = settings.fom_function
        else:
            fom_function = lambda x: True
                
        if hasattr(settings, 'simulation_folder'):
            simulation_folder = settings.simulation_folder
        else:
            simulation_folder = ''
            
        if simulation_folder == '':
            simulation_folder = current_folder
            
        if hasattr(settings, 'hide_gui'):
            hide_gui = settings.hide_gui
        else:
            hide_gui = False
        
        if hasattr(settings, 'simulator_restart'):
            self._simulator_restart = settings.simulator_restart
        else:
            self._simulator_restart = True
        
        if hasattr(settings, 'simulation_builder'):
            self.simulation_builder = settings.simulation_builder
        else:
            self.simulation_builder = None
        
        # File name to save fsp file before simulation
        self._file_name=file_name
        
        # Create the simulator #
        ########################
        self.sim = LumericalFDTD(simulation_folder, hide_gui)
        self._sim_initialized = False
        
        # Define the problem #
        ######################      
        # Geometry
        self.geometry = LumericalGeometryObject(geometry_function, initial_parameters,
                                                self.sim, parameters_name=self.parameters_name)
        
        # FOM
        self.fom = LumericalFomObject(fom_function, self.sim)
        

            
    def run(self, param=None):
        """ Run a simulation and return the results
        
        Parameters
        ----------
        param: array, None
            The input parameters used for the simulation. If None, the simulation
            run without updtaing the geometry

        Returns
        -------
        res: array
            The simulation result
        used_parameters: list
            The parameters used for the simulation (useful if param=None)
        
        """
        
        # Error handling is not done here but in the sampler class
        self._initialize_simulation()
        if param is not None:
            self.geometry.update(param = np.array(param).flatten(), update = 1, push_change = True)
        self.sim.run(self._file_name)
        res = self.fom.get()
        if self._simulator_restart:
            self.close_simulation()
        
        # Return the simulation result and the used parameters (useful
        # if param=None)
        return (res, self.geometry.parameters)
    
    
    def update_geometry(self, param=None):
        """ Update the simulation geometry
        
        Parameters
        ----------
        param: array, None
            The new parameters. If param = None the geometry is not updated

        Returns
        -------
        res: bool
            True if geometry has been updated
        
        """
        if param is not None:
            self.geometry.update(param=np.array(param).flatten(), update = 1, push_change = False)
            res = True
        else:
            res = False

        return res
    
    def change_fom(self, fom_function, fom_name='fom'):
        """
        Change the figure of merit 
        
        Parameters
        ----------
        fom_function: Fom object
            Object to retrive simulation results
        fom_name: string
            Label for the returned simulation result

        """
        
        self.fom = LumericalFomObject(fom_function, self.sim)
        self.fom_name = fom_name
        print("New figure of merit: " + fom_name)
        
    
    def close_simulation(self):
        """ Close simulation interface """
        
        if self._sim_initialized == True:
            
            # Close the simulator interface
            self.sim.close()
        
            # Flag not initialized
            self._sim_initialized = False
 
    
    def _initialize_simulation(self):
        """ Initialize simulation interface """
        
        if self._sim_initialized == False:
            # simulation initilization
            self.sim.initialize(self.simulation_builder)
            
            # Initialize the geometry (add it to the simulation)
            self.geometry.initialize()
            
            # Initialize fom (required for lumopt)
            self.fom.initialize()
            
            # Flag initialized
            self._sim_initialized = True
    

        


        
