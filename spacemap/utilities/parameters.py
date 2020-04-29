import datetime

class Parameters:
    """
        Empty class to store parameters
    """
        
    pass

class Settings(Parameters):
    """
        Class for settings structuring
        
        Attributes
        ----------
        general : Parameters
            General spacemap settings
        study: Parameters
            Study-specific settings
        sampler: Parameters
            sampler-specific settings
        dimensionality_reduction: Parameters
            Settings for dimensionality reduction

    """
        
    def __init__(self):
                
        self.general = Parameters()
        self.general.suffix = 'SpaceMapping_' + datetime.datetime.now().strftime("%Y_%m_%d_%I_%M")
        self.general.comments = ''
        self.general.autosave = True
        
        self.study = Parameters()
        self.study.type = 'LumericalFDTD'
         
        self.sampler = Parameters()
        self.sampler.type = 'random-lumopt' 

        self.dimensionality_reduction = Parameters()
        self.dimensionality_reduction.type = 'pca'
