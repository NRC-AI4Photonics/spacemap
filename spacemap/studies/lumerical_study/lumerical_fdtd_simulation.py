import lumapi
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
import os

class LumericalFDTD():
    """
    Class to abstract the interface with Lumerical FDTD

    Parameters
    ----------
    working_dir: sting
        Simulation directory
    hide_gui: bool
        If true, hide the GUI during simulation
    
    Attributes
    ----------
    fdtd: api handle
        Handle to the active FDTD simulator
    hide_gui: bool
        If true, hide the GUI during simulation
    """
    
    def __init__(self, working_dir, hide_gui):
        self._working_dir = working_dir
        self.hide_gui = hide_gui
        
   
    def __del__(self):
        self.close()

    
    def initialize(self, build_simulation):
        """ Launches FDTD CAD and stores the handle """
        # lumopt objects expect self.fdtd to be a lumapi.FDTD handle
        self.fdtd = lumapi.FDTD(hide = self.hide_gui)
        self.fdtd.cd(self._working_dir)
        
        # build the simulation
        self.execute(build_simulation)

    

    def execute(self, obj):
        """
        Interact with the simulator
        
        Params
        -------
        obj: function handle, string
            The code to be executed by the simulator
        
        """
        
        if callable(obj):
            try:    
                # assume it is a lumapi code, execute passing the lumapi handle
                exec_res = obj(self.fdtd)
            except:
                # does not take arguments, assume it returns a string to evaluate
                exec_res = self.execute(obj())
            finally:
                return exec_res
        elif isinstance(obj, str):
            # check if the string is a file
            full_path_file = os.path.abspath(self._working_dir+obj)
            if not os.path.isfile(full_path_file):
                # Evaluate a Lumerical script if a string is passed
                self._script_eval(obj)
            else:
                #assume a file name is passed: load it
                self._load_simulation_file(full_path_file)
            
      

    def _script_eval(self, script):
        """ Evaluate a script """
        self.fdtd.eval(script)
        
    def _load_simulation_file (self, file):
        """ Load a simulation file """ 
        
        # Assumes file is either an absolute path or accessible in the 
        # current directory
        
        if '.fsp' in file and os.path.isfile(file):
           self.fdtd.load(file)
        elif '.lsf' in file and os.path.isfile(file):
            script_str = load_from_lsf(file)
            self._script_eval(script_str)
        else:
            raise UserWarning('Input file must be either .fsp or .lsf')
               
    
    def run(self, name):
        """
        Saves simulation file and run the simulation
        
        Params
        -------
        name: string
            Name for the fsp file
        
        """
             
        self.fdtd.save('{}'.format(name))
        self.fdtd.run()
    
    def close(self):
        """ Close the simulator and delete the handle """ 
        
        if hasattr(self, 'fdtd'):
            self.fdtd.close()
            # Must clear the object to remove pointers or pickle does not work
            del self.fdtd

