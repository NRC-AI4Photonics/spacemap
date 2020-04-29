from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.geometries.parameterized_geometry import ParameterizedGeometry
import numpy as np

class LumericalGeometryObject:
    """
        Class to abstract different types of geometry definitions that can work
        with Lumerical FDTD interface
        
        Parameters
        ----------
        geom_obj: function, object
            The object defining the geometry
        initial_param: array
            Initial set of parameters (only used if geom_obj is a function handle)
        ha: object
            Handle for Lumerical FDTD
        parameters_name: list, None
            Names to use for the parameters of the geometry
            
        Attributes
        ----------
        geometry: function, object
            The geom_obj passed to init
        parameters: numpy array
            The current parameters of the geometry
        parameters_size: int
            The number of parameters
        parameters_name: array, None
            1D array of strings with parameters names
    """
    
    def __init__(self,geom_obj, initial_param, ha, parameters_name):
        
        # Simulation handle
        self._ha = ha

        if isinstance(geom_obj,FunctionDefinedPolygon):
            self.geometry = geom_obj
            self.parameters = np.array(self.geometry.get_current_params())
            self.parameters_size = np.size(self.parameters)
            self._geometry_type = 'FunctionDefinedPolygon'
        elif isinstance(geom_obj,ParameterizedGeometry):
            self.geometry = geom_obj
            self.parameters = np.array(self.geometry.get_current_params())
            self.parameters_size = np.size(self.parameters)
            self._geometry_type = 'ParameterizedGeometry'
        else:
            # Function can include either a lumapi code or return a string
            # (lumerical script to execute). This cases are handled by the 
            # Simulation class 
            self._geometry_function = geom_obj
            # real geometry is created once the first set of parameters is provided
            self.geometry = lambda: self._geometry_function(np.array(initial_param))
            self.parameters = np.array(initial_param)
            self.parameters_size = np.size(self.parameters)
            self._geometry_type = 'function'
            
        # Store parameters names
        if parameters_name == None:
            names = list()
            for idx in range(0,self.parameters_size):
                names.append('Var' + str(idx))
            self.parameters_name = names
        else:
            self.parameters_name = parameters_name

    def initialize(self):
        """ Initialize the geometry """
        
        self.update(self.parameters,update = 0, push_change = True)
    
        
    def update(self, param, update = 1, push_change = False):
        """
        Update the geometry
        
        Parameters
        ----------
        param: array
            New set of parameters
        update: int
            Either 1 or 0, used with lumopt geometries to flag if geometry is new or an update
        push_change: bool
            If False, the new parameter set is stored but the geometry is not actually updated
        
        """
        
        # Check if parameters are passed
        param_flag = not np.array(param==None).any()
        
        if self._geometry_type == 'function':
            # Update stored parameters if param_flag=True
            if param_flag:
                self.geometry = lambda: self._geometry_function(np.array(param))
                self.parameters = np.array(param)
            
            if push_change:
                self._ha.execute(self.geometry)
        elif self._geometry_type == 'FunctionDefinedPolygon':
            # Update stored parameters if param_flag=True
            if param_flag:
                self.geometry.update_geometry(np.array(param))
            
            new_params = self.geometry.get_current_params()
            self.parameters = np.array(new_params)
            
            if push_change:
                self.geometry.add_geo(self._ha, new_params, update)
                
        elif self._geometry_type == 'ParameterizedGeometry':
            # Update stored parameters if param_flag=True
            if param_flag:
                self.geometry.update_geometry(np.array(param), self._ha)
            
            new_params = self.geometry.get_current_params()
            self.parameters = np.array(new_params)
            
            if push_change:
                self.geometry.add_geo(self._ha, new_params, update)
            
        self.parameters_size = np.size(self.parameters)
        
        

            