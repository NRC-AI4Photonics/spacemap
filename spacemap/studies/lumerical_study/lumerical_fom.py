from lumopt.figures_of_merit.modematch import ModeMatch

class LumericalFomObject:
    """
    Class to abstract different types of figure of merit definitions that
    can work with Lumerical FDTD
        
    Parameters
    ----------
    fom_obj: function, object
        MoeMatch object or function to extract simulation results
    ha: object
        Handle for Lumerical FDTD
    
    Attributes
    ----------
    fom: function, object
        The fom_obj passed to init       
    """
    
    def __init__(self, fom_obj, ha):
        
        self._ha = ha
        if isinstance(fom_obj,ModeMatch):
            self.fom = fom_obj
            self._fom_type = 'ModeMatch'
        else:
            # Function can include either a lumapi code or return a string
            # (lumerical script to execute). This cases are handled by the 
            # Simulation class
            self.fom = fom_obj
            self._fom_type = 'function'
    
    
    def initialize(self):
        """ Initialize the figure of merit """
        
        # lumopt.figures_of_merit.modematch object need initialization and
        # forward setting. h is a spacemap.utilities.simulation object
        if self._fom_type == 'ModeMatch':
            self.fom.initialize(self._ha)
            self.fom.make_forward_sim(self._ha)
    
    
    def get(self):
        """
        Get the figure of merit 
        
        Returns
        -------
        res: array
            The simulation result
        
        """
        
        if self._fom_type == 'function':
            # It is a function with no parameters, just run it
#            return self.fom(self.ha.fdtd.handle)
            return self._ha.execute(self.fom)
        elif self._fom_type == 'ModeMatch':
            # It is a lumopt.figures_of_merit.modematch object
            return self.fom.get_fom(self._ha)