import logging
import yaml
from dataclasses import dataclass
import numpy as np
import os

@dataclass
class PSOparameters():
    maxIter: int  # max iterations in PSO search
    nPop:int  # swarm size
    inertia:float   # inertia (for velocity)
    c1:float  # personal acceleration coefficient
    c2:float  # social acceleration coefficient
    tol:float  # tolerance - stop PSO algorithm if gbest is lower than this
    showIter:bool  # show iteration information
    varaibles: np.array # Varaibles that shoudl be estimated with PSO
    num_varaibles: int # total number of varaibles in PSO
    wDelta: float # dynamic inertia parameter
    wMin: float #  dynamic inertia parameter
    wMax: float #  dynamic inertia parameter
    var_lim_min: dict # Multipliers for min var limit in set pso problem
    var_lim_max: dict # Multipliers for max var limit in set pso problem
    first_level_objectiv: str # Wich first level objectiv should be used
    velocity_start_quota: int # States how much the diff between the max and min paramter should be changed for ideal velocity calcaultion
    eq_additional_constraints: str # additional cosntraints for Max power produktion
    reserve_method: str # Reserve Method to hold back power for primary reserver
    ramp_short: int # Ramp for shorter time
    ramp_long: int # Longer Ramp

def load_yaml_file(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

def save_yaml_file(filename, data):
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def set_config(
        area, run_setup, config_file_name, overwrite_files = False, category:int = 0, conf_filname:str='main_PSO.log', folder:str=None, year=''):
    """Set the config for 

    Args:
        area (_type_): _description_
        run_setup (_type_): _description_
        config_file_name (_type_): _description_
        category (int, optional): _description_. Defaults to 0.
        conf_filname (str, optional): _description_. Defaults to 'main_PSO.log'.
        folder (str, optional):'select' means path select, None is the test configs, otherwise the path. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Configure the root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        filename=conf_filname,  # Specify the log file name
        filemode='w'  # Set the file mode ('w' for overwrite, 'a' for append)
    )
    if not folder:
        folder_name = 'configs/'

    else:
        folder_name = folder + "\\" + run_setup + "\\"

    with open(folder_name +'conf\\'+ config_file_name + '.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    param_pso = PSOparameters(
        maxIter=config['para_PSO']['max_inter'],
        nPop=config['para_PSO']['population'],
        inertia=config['para_PSO']['intertia'],
        c1=config['para_PSO']['c1'],
        c2=config['para_PSO']['c2'],
        tol=config['para_PSO']['tol'],
        showIter=config['para_PSO']['show_iter'],
        varaibles=config['para_PSO']['variables'],
        num_varaibles = len(config['para_PSO']['variables']),
        wDelta=config['para_PSO']['wDelta'],
        wMin=config['para_PSO']['wMin'],
        wMax=config['para_PSO']['wMax'],
        var_lim_max=dict(config['para_PSO']['var_lim_max']),
        var_lim_min=dict(config['para_PSO']['var_lim_min']),
        first_level_objectiv=config['para_PSO']['first_level_objectiv'],
        velocity_start_quota = 1, # 3.5
        eq_additional_constraints = config['eq_additional_constraints'],
        reserve_method = None,
        ramp_short = int(config['para_general']['small_ramp']),
        ramp_long = int(config['para_general']['long_ramp']),
    )
    if year:
        year = year + '_'
        area_year = area + '_' + year 

    if not 'file_location' in config:
        config['file_location'] = {}
        config['file_location']['output'] = folder_name + 'Equivalent solutions\\'
        config['file_location']['input'] = folder_name + 'Input\\'
    
    if not 'detailed' in config['file_location']:
        config['file_location']['detailed'] = config['file_location']['output'] + "\\Detailed model solutions\\"
        config['files']['M_file'] = config['file_location']['input'] + config['files']['M_file']
        config['files']['basic_files'] = config['file_location']['input'] +'Historical Production\\' + area + '_' +config['files']['basic_files']
    
    if not 'graphs' in config['file_location']:
        config['file_location']['graphs'] = folder_name + 'graphs\\'
    else:
        config['file_location']['graphs'] = config['file_location']['graphs'] + run_setup + '\\graphs\\'

    config['files']['train_price_file'] = config['file_location']['input']+ 'Price\\' + area_year + config['files']['train_price_file']
    if 'inflow_file' in config['files']:
        config['files']['inflow_file'] = config['file_location']['input'] + config['files']['inflow_file']

    config['files']['day_file'] = config['file_location']['input'] + year + config['files']['day_file']
      

    config['para_general']['category'] = category

    config['area'] = area

    config['output_filename_suffix'] = (config['output_filename_suffix'] + area + '_cat_'+ str(category)
                                        + '_iter' + str(param_pso.maxIter) + '_pop' + str(param_pso.nPop))

    if not os.path.exists(config['file_location']['output']):
        raise FileNotFoundError('Outpiut Folder not exists')
    if not os.path.exists(config['file_location']['graphs']):
        raise FileNotFoundError('Graph Folder not exists')   
    

    if 'ext_inflow_file' in config['files']:
        config['files']['ext_inflow_file'] = config['file_location']['input'] + 'Inflow\\' + area_year + config['files']['ext_inflow_file']

    if 'eq_inflow' not in config['para_general']:
        config['para_general']['eq_inflow'] = 'calculation'

    if 'real_production' in config['files']:
        config['files']['real_production'] = config['file_location']['input'] +'Historical Production\\' + area_year + config['files']['real_production']

    if not overwrite_files:
        config = check_existing_run(config)
        
    return config, param_pso

def check_existing_run(config):
    suffix_counter = 1
    while os.path.exists(config['file_location']['output'] + "\\Equivalent solutions\\" + config['output_filename_suffix'] + "config.yaml"):
        config['output_filename_suffix'] = config['output_filename_suffix'] + f'_{suffix_counter}'
        suffix_counter += 1

    return config