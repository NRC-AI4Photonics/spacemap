import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
import itertools

class DimensionalityReductionPCA:
    """
        Class to handle dimensionality reduction (DR)
        
        Parameters
        ----------
        settings: Parameters
            List of settings for the dimensionality reduction
            
            settings.type: str (default: 'pca')
                DR algorithm
            settings.n_components: int (default 2)
                Number of components of the reduced space (for PCA)
            settings.scale: bool (default False)
                Scale training data standard deviation before DR
            
        Attributes
        ----------
        n_components: int
            Number of components of the reduced space (for PCA)
        training_data: array
            Data used for training
        projected_training_data: array
            Training data projected on the DR model
        model: object
            DR model (for PCA, sklearn PCA)
    """
    
    def __init__(self, settings):
        
        # Read settings #
        #################
    
        if hasattr(settings, 'type'):
            self._type = settings.type
        else:
            self._type = 'pca'
            
        if hasattr(settings, 'n_components'):
            self.n_components = settings.n_components
        else:
            self.n_components = 2
        
        # n_components could change during execution, store the initial setting
        self._init_n_components = self.n_components
            
        if hasattr(settings, 'scale'):
            self._scale = settings.scale
        else:
            self._scale = False
        

        
    def train_model(self, training_data, disp_result=True):
        """
        Train the DR model
        
        Parameters
        ----------
        training_data: array
            Input data for the training
        disp_result: bool
            Display some info after model is trained
        
        """
        if self._type == 'pca':
            self._run_pca(training_data, disp_result)
    
    def invert_model(self,data):
        """
        
        Invert the DR model
        
        Parameters
        ----------
        data: array
            Data to be projected back in the original space

        Returns
        -------
        result: array
            Back-projected data

        
        """
        if self._type == 'pca':
            return self._invert_pca(data)
    
    def project (self,data):
        """
        
        Project new data on the DR model
        
        Parameters
        ----------
        data: array
            Data to be projected in the reduced space

        Returns
        -------
        result: array
            The projected data
        
        """
        if self._type == 'pca':
            return self._project_pca(data)
   
    
    def subspace_mesh(self, distance = None, n_points = None, boundaries = None):
        """        
        Create a mesh on the reduced space.
        
        Parameters
        ----------
        distance: float, array, None
            Manhattan distance (in the original space, not in the reduced space!)
            between points. If scalar, the same distance is used for all the
            reduced dimensions
        n_points: int, array, None
            Number of sampling points. If scalar, the same number of points is
            used for all the reduced dimensions
        boundaries: array(2,n_components), None
            The lower (boundaries[0]) and upper (boundaries[1]) boundary of the
            region to be sampled in the reduced space. Boundaries are specified
            in units of the reduced space. Default: +-5 times the standard
            deviation of the projected training data.
        

        Returns
        -------
        mapping_grid: list
            List of the sampled points, in the original space    
        mapping_projected_grid: list
            List of the sampled points, in the reduced space
        mapping_normalized_projected_grid: list
            List of the sampled points, in the reduced space but normalized
            such that the distance between two consecutive points in each 
            dimension is the same (that specified in the parameters, if
            available)
            
        
        Notes
        -----        
        Either distance or n_points must be different from None
        
        """
               
        # We can receive either distance or n_points. Both can be scalar or vectors
        if n_points == None:
            if distance == None:
                # Cannot be both None
                raise ValueError('Subspace meshing cannot be done, both distance and number of points are not specified')
            else:
                # We have the distance, must find the number of points
                if isinstance(distance,np.ndarray):
                    if len(distance)==1:
                        dist_vector = np.ones(self.n_components,)*distance
                    else:
                        dist_vector = distance
                else:
                    #assume is a scalar
                    dist_vector = np.ones(self.n_components,)*distance
            
                n_points_vector = np.zeros(self.n_components,dtype="int")
                
                flag_find_n_points = True
                
        else:
            # We have the number of point, we have to find the distances
            if isinstance(n_points,np.ndarray):
                if len(n_points)==1:
                    n_points_vector = np.ones(self.n_components,dtype="int")*n_points
                else:
                    # be sure we have a numpy array of int
                    n_points_vector = np.array([int(i) for i in n_points])
            else:
                #assume is a scalar
                n_points_vector = np.ones(self.n_components,dtype="int")*n_points
            
            dist_vector = np.zeros(self.n_components)
            
            flag_find_n_points = False
         
        # Define the boundaries
        if boundaries != None:
            lower_bound = boundaries[0]
            upper_bound = boundaries[1]
            
        else:
            samples_std = self.projected_training_data.std(axis=0)
            samples_mean = self.projected_training_data.mean(axis=0) #this should be zero!
        
            lower_bound = samples_mean - 5*samples_std
            upper_bound = samples_mean + 5*samples_std
 
        mapping_lists = list()
        mapping_lists_normalized = list()  
        for i in range(0,self.n_components):
            point_a_projected = np.zeros(self.n_components)
            point_a_projected[i] = lower_bound[i]
            point_a = self.invert_model(point_a_projected)
            point_b_projected = np.zeros(self.n_components)
            point_b_projected[i] = upper_bound[i]
            point_b = self.invert_model(point_b_projected)
                
            ab_dist = np.sum(np.abs(point_a-point_b))
            if flag_find_n_points:
                n_points_vector[i] = int(np.ceil(ab_dist/dist_vector[i]) + 1)
            else:
                dist_vector[i] = ab_dist/(n_points_vector[i]-1)
        
            samp = np.linspace(lower_bound[i], upper_bound[i], n_points_vector[i])
            mapping_lists.append(samp)
            if len(samp) > 0:
                projected_dist = samp[1]-samp[0]
                mapping_lists_normalized.append(samp/projected_dist*dist_vector[i])
            else:
                # If we have only one sample it is at zero (origin) of the projected axis
                mapping_lists_normalized.append(samp)
            
            
            
        mapping_projected_grid = list(itertools.product(*mapping_lists))
        mapping_normalized_projected_grid = list(itertools.product(*mapping_lists_normalized))
        mapping_grid = self.invert_model(mapping_projected_grid)
        
        return mapping_grid, mapping_projected_grid, mapping_normalized_projected_grid
    
    def reset_model(self):
        """ Clear the model and training data, bringing back the class to the init state """
        
        if self._type == 'pca':
            self._reset_pca()
        
    def _reset_pca(self):
        """ Clear pca model and training data, bringing back the class to the init state """
        
        del self.training_data
        del self._scaler
        del self.model
        del self.projected_training_data
        self.n_components = self._init_n_components
        
    
    def _run_pca(self, training_data, disp_result):
        """ Run scaler and PCA algorithm """
        
        # Store training data
        self.training_data = np.array(training_data)
        
        # Scale the samples
        if self._scale:
            self._scaler = StandardScaler()
        else:
            self._scaler = StandardScaler(with_std = False)
        scaled_params = self._scaler.fit_transform(self.training_data)
        
        self.model = PCA(n_components = self.n_components)
        
        self.projected_training_data = self.model.fit_transform(scaled_params)
        self.n_components = len(self.model.components_)
           
        if disp_result:
            print("Expained variance {}, sum {}".format(self.model.explained_variance_ratio_,sum(self.model.explained_variance_ratio_)))
        
    def _invert_pca(self,data): 
        """ Invert PCA and scaling """
        return self._scaler.inverse_transform(self.model.inverse_transform(data))
    
    def _project_pca(self,data):
        """ Project new data """
        return self.model.transform(self._scaler.transform(data))