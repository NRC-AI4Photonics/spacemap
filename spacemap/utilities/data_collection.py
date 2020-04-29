import numpy as np
import csv

class Sample:
    """
        Class to store a single simulation sample
        
        Parameters
        ----------
        parameters: array
            Set of parameters. 1D list or numpy array of length (n)
        simulation_name: str
            Name of the simulation stage that generated the result
        result: list
            The results generated by the simulation
        result_name: str
            A name for the function that generated the result
        
        Attributes
        ----------
        parameters: numpy array
            The set of parameters
        parameters_size: int
            Number of parameters
        simulation_name: str
            Name of the simulation stage that generated the result  
        result:list
            The results generated by the simulation
        result_size: int
            The lenght of the 'result' list
        result_name: str
            A name for the function that generated the result
    """
    def __init__(self, parameters, simulation_name, result, result_name):
        self.parameters = np.array(parameters).flatten()
        self.parameters_size = self.parameters.shape[0]
        
        self.simulation_name = simulation_name
        
        #if result is not a list, make it a list
        if (type(result)!=list):
            result_to_store = [result]
        else:
            result_to_store = result
        self.result = result_to_store
        self.result_size = len(self.result)
        self.result_name = result_name
        
        
    

class DataCollection:
    """
        Class to store the information about the optimization,
        parameters, geometry, and figure of merit.
        
        Attributes
        ----------
        sample: list
            List of 'Sample' object, one for each entry
        sample_num: int
            Number of entries   
        current_sample: obj
            Points to sample[-1]
    """
    
    def __init__(self):
        self.sample = list()
        self.sample_num = 0
        self.current_sample = None

        
    def __iter__(self):
        self._n_iteration = 0
        return self
    
    def __next__(self):
        if self._n_iteration < self.sample_num:
            self._n_iteration += 1
            return self.sample[self._n_iteration -1]
        else:
            raise StopIteration

        
    def add_sample(self, parameters, simulation_name, result, result_name):
        """        
        Add a sample to the collection
        
        Parameters
        ----------
        parameters: array
            Set of parameters. 1D list or numpy array of length (n)
        simulation_name: str
            Name of the simulation stage that generated the result
        result: list
            The results generated by the simulation
        result_name: str
            A name for the function that generated the result
        

        Returns
        -------
        idx: int
            The index of the added sample
        
        """
        
        self.sample.append(Sample(parameters, simulation_name, result, result_name))
        self.current_sample = self.sample[-1]
        self.sample_num = len(self.sample)
        
        return self.sample_num-1 #index of the added sample
        
    def filter_simulation(self, result_name, result_index = 0, simulation_name = None, lower_bound = None, upper_bound = None):
        filtered_parameters = list()
        filtered_result = list()
        filtered_idx = list()
        
        # Convert to numpy arrays, simpler conditions management
        if lower_bound is not None:
            lb = np.array(lower_bound)
        else:
            lb = None
        if upper_bound is not None:
            ub = np.array(upper_bound)
        else:
            ub = None
        
        for i in range(0,self.sample_num):
            if (self.sample[i].result_name == result_name) and ((simulation_name is None) or (self.sample[i].simulation_name in simulation_name)):
                # Check boundaries
                res = np.array(self.sample[i].result[result_index])
                if type(result_index) is int:
                    if ((lb == None or (res > lb).all()) and
                        (ub == None or (res < ub).all())):
                        
                        filtered_parameters.append(self.sample[i].parameters)
                        filtered_result.append(self.sample[i].result)
                        filtered_idx.append(i)
                else:
                    # It is a list and also boundaries are two lists
                    to_include = list()
                    for res_idx in range(0, len(result_index)):
                        res = np.array(self.sample[i].result[res_idx])
                        if ((lb == None or lb[res_idx] == None or (res > lb[res_idx]).all()) and
                            (ub == None or lb[res_idx] == None or (res < ub[res_idx]).all())):
                        
                            to_include.append(True)
                        else:
                            to_include.append(False)
                    
                    if all(to_include):
                        filtered_parameters.append(self.sample[i].parameters)
                        filtered_result.append(self.sample[i].result)
                        filtered_idx.append(i)
        
        return (filtered_parameters, filtered_result, filtered_idx)
    
    def is_sample(self, parameters, result_name):
        sample_found = False
        idx = None
     
        for i in range(0,self.sample_num):
            if len(parameters) == self.sample[i].parameters_size:
                if all(self.sample[i].parameters == parameters) and (self.sample[i].result_name is result_name):
                    sample_found = True
                    idx = i
                    break
                
        return sample_found, idx
    
    def load_from_csv(self, file_name, parameters_size):
        """ Load data from csv. Assumes the first parameter_size colums are parameters,
            then simulation name, than result name, while all the other columns are different results """
        
        with open(file_name) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                simulation_name = str(row[0:1])[2:-2] # Remove brakets and quotation marks 
                result_name = str(row[1:2])[2:-2] # Remove brakets and quotation marks 
                params = np.float64(row[2:parameters_size+2])
                res = np.float64(row[parameters_size+2:])
                self.add_sample(parameters = params, simulation_name = simulation_name, result = res, result_name = result_name)
                
    def export_to_csv(self, file_name):
        """ Export the database to a csv file """
        
        with open(file_name, 'w', newline='') as csvDataFile:
            csvWriter = csv.writer(csvDataFile, delimiter = ',')

            for i in range(0,self.sample_num):
                data = list()
                data.append(self.sample[i].simulation_name)
                data.append(self.sample[i].result_name)
                data.extend(self.sample[i].parameters.tolist())
                data.extend(self.sample[i].result)                                 
                csvWriter.writerow(data)

        
        
        
        
        