import numpy as np
from .random_local import RandomLocal
from .random_lumopt import RandomLumopt


class Sampler:
    """
        Basic class defining the the sampler

    """
    
    @staticmethod
    def get_sampler(settings, study):
        """ 
        Return the proper sampler type 
        
        Parameters
        ----------
        settings: Parameters
            List of settings appropriate for the selected sampler
        study: object
            The study object to be used by the sampler
        
        Returns
        ----------
        sampler: object
            The sampler of the requested type
        
        """
        
        if settings.type == 'random-local':
            return RandomLocal(settings=settings,study=study)
        elif settings.type == 'random-lumopt':
            return RandomLumopt(settings=settings,study=study)
        
    