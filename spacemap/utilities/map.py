class Map:
    """
    Class to store mapping results
    
    Parameters
    ----------   
    model: obj
        The dimensionality reduction model
    training_data_idx: list
        The index in the database (data_collection class) of the data used for the training
    projected_grid: list
        List of the sampled points, in the reduced space 
    normalized_projected_grid: list
        List of the sampled points, in the reduced space but normalized 
        such that the distance between two consecutive points in each 
        dimension is the same
    sim_idx: list
        The index in the database (data_collection class) of the simulations that are part of the map
        (includes both parameters in the original space and simulation results)
    """
    
    def __init__(self, model, training_data_idx, projected_grid, normalized_projected_grid,
                 sim_idx):
        
        self.model = model
        self.training_data_idx = training_data_idx
        self.projected_grid = projected_grid
        self.normalized_projected_grid = normalized_projected_grid
        self.sim_idx = sim_idx
        