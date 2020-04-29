import numpy as np
from .pca import DimensionalityReductionPCA


class DimensionalityReduction:
    """
        Basic class defining the dimensionality reduction

    """
    
    @staticmethod
    def get_dm(settings):
        """ 
        Return the proper dimensionality reduction object
        
        Parameters
        ----------
        settings: Parameters
            List of settings appropriate for the selected dimensionality reduction method
        
        Returns
        ----------
        dm: object
            The dimensionality reduction object
        
        """
        
        if settings.type == 'pca':
            return DimensionalityReductionPCA(settings=settings)

        
    