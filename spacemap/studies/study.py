import numpy as np
from .lumerical_study.lumerical_fdtd_study import LumericalFDTDStudy


class Study:
    """
    Basic class to instantiate the study
    
    """
    
    @staticmethod
    def get_study(settings, file_name):
        """ 
        Return the proper study type 
        
        Parameters
        ----------
        settings: Parameters
            List of settings appropriate for the selected study
        file_name: str
            An available file name to save service files in the study (if needed)
        
        Returns
        ----------
        study: object
            The study of the requested type
        

        """
        
        if settings.type == 'LumericalFDTD':
            return LumericalFDTDStudy(settings, file_name)
        
    