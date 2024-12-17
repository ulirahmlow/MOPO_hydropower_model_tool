import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScenarioData:
    days : int = None
    dwnCapacity : int = None            # Required down-regulation capacity
    hours : int = None                  # Number of hours in each scenario
    initialEnergy : float = None        # Initial energy in the system
    M0share : pd.DataFrame = None       # Relative energy content at the start (row 1) and end (row 2) of the planning period
    MTminShare : float = None           # Min relative energy content at the end of the planning period
    price : pd.Series = None            # Price matrix
    scenarios : int = None              # Total number of scenarios
    upCapacity : int = None             # Required up-regulation capacity
    max_content_energy: float = None    # Max reservoar capacity total, if available
    max_historical_production: float = None # Max historical production, if available
    min_historical_production: float = None # Min historical production, if available

def read_scenario_data(conf):

    scenario_data = ScenarioData()
    scenario_data.price = pd.read_csv(conf['files']['train_price_file'],delimiter=';',index_col=[0]).reset_index(drop=True)
    scenario_data.days = pd.read_csv(conf['files']['day_file'],delimiter=';')
    hours, scen = scenario_data.price.shape
    scenario_data.hours = hours # hours in each scenario
    scenario_data.scenarios = scen # nbr scenarios
    M_temp = pd.read_csv(conf['files']['M_file'],delimiter=';')
    scenario_data.M0share = M_temp.iloc[:-1, 1:]
    scenario_data.MTminShare = M_temp.iloc[-1,1:] #§ only one value for all

    # If we have a file with the max content, we read it
    if 'basic_files' in conf['files']:
        initial_parameter = pd.read_csv(conf['files']['basic_files'],index_col=[0],delimiter=';')
        scenario_data.max_content_energy = initial_parameter.loc[conf['area'],'max_M(MWh)']
        scenario_data.initialEnergy = scenario_data.max_content_energy * scenario_data.M0share.iloc[0,0]
        scenario_data.max_historical_production = initial_parameter.loc[conf['area'],'max_P(MWh)']
        scenario_data.min_historical_production = initial_parameter.loc[conf['area'],'min_P(MWh)']
        logger.info(f"Initial Parameters read from file for area:{conf['area']}")

    else:
        scenario_data.max_content_energy = 10000000000 #initlize value 
        scenario_data.initialEnergy = None
        logger.warning(f"Max content set to max :{scenario_data.max_content_energy}")

    if 'real_production' in conf['files']:
        scenario_data.real_production = pd.read_csv(conf['files']['real_production'],delimiter=';',index_col=[0]).reset_index(drop=True)
        
    return scenario_data