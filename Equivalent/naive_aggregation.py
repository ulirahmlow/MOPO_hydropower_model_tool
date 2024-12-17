import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

@dataclass
class EqModel():
    alpha : pd.DataFrame = None # Vector defining M0 in each reservoir
    dwnAd : pd.DataFrame = None          # Matrix of downstream reservoirs (including itself)
    inflow : pd.DataFrame = None         # Local inflow to each station
    Mmax : pd.DataFrame = None           # Max reservoir content, for all reservoirs
    Mmin : pd.DataFrame = None           # Min reservoir content, for all reservoirs
    mu : pd.DataFrame= None              # Marginal production equivalent for each plant and segment
    nbr_stations : int = None    # Number of stations in Equivalent
    Qmax : pd.DataFrame = None         # Min discharge in each plant and segment
    Qmin : pd.DataFrame = None           # Max discharge in each plant and segment
    ramp1h : pd.DataFrame = None         # Limit on ramping for 1 hour
    ramp4h : pd.DataFrame =None         # Limit on ramping for 4 hour
    ramp3h : pd.DataFrame =None         # Limit on ramping for 3 hour
    Smin : pd.DataFrame = None            # Minimum spill (instead of Smax?)
    segments : int = None       # Number of segments in mu
    scenarios : int = None      # Number of inflow scenarios
    upAq : pd.DataFrame = None           # Matrix of upstream reservoirs
    delay: pd.DataFrame = None           # Delay between two reservoirs
    indices: dict = None # Indicies for simulation
    indices_with_segments: dict = None  # Indicies for Simulation
    indices_thermal: dict = None        # Indicies for Thermal Power Production
    Q_break: np.array = None            # Set the paramater if the eq has segments and when how
    delay: pd.DataFrame = None          # Delay bettwen two reservoirs to flow
    indices_only_times: dict = None     # Indicies for just time and scenarios   
    delta_max_prod: float = None        # delta between max and production 
    inflow_multiplier: float = None      # inflow_multiplier to change inflow

logger = logging.getLogger(__name__)

def create_ramping_data(orig_system, conf):
    t1 = conf['para_general']['small_ramp']
    t2 = conf['para_general']['long_ramp']
    tim, scen = orig_system.power.shape # total number of hours and scenarios
    # ramping for t1 and t2:
    prev1 = 0
    prev2 = 0 # initialize
    for w in orig_system.power:
        for t in range(t2, tim):
            temp1 = abs(orig_system.power.loc[t,w] 
                        - orig_system.power.loc[t-t1,w])
            temp2 = abs(orig_system.power.loc[t,w] 
                        - orig_system.power.loc[t-t2,w])
            if temp1 > prev1:
                prev1 = temp1
            # end if temp ramp1 greater than previous greatest ramp1
            if temp2 > prev2:
                prev2 = temp2
            # end if temp ramp4 greater than previous greatest ramp4

    return prev1, prev2

def create_eq_model(scenario, conf) -> EqModel:
    tim, scen = scenario.price.shape # total number of hours and scenarios
    eq_init = EqModel()
    eq_init.nbr_stations = conf['para_general']['stations']   # number of aggregated stations
    eq_init.segments = conf['para_general']['segments']       # segments in production function

    (eq_init.indices_with_segments, 
     eq_init.indices, 
     eq_init.indices_only_times) =  create_indicies(eq_init.nbr_stations, tim, scen, eq_init.segments, scenario)
    eq_init.mu = [seg for seg in conf['para_general']['prod_seg_init']]
    eq_init.upAq = pd.DataFrame(conf['para_general']['up_aq'],dtype=int) # upsteam stations
    eq_init.alpha = pd.DataFrame({'alpha':conf['para_general']['alpha']},dtype=float) # share of energy content in each reservoir
    eq_init.dwnAd = pd.DataFrame(conf['para_general']['down_ad'],dtype=int)  # downstream stations
    eq_init.Qmax = pd.DataFrame([10000],dtype=float) # Max discharge for each segment
    eq_init.Qmin = pd.DataFrame([1],dtype=float) # Min discharge for each segment
    eq_init.ramp1h = pd.DataFrame({'ramp1h':[100000]},dtype=float)
    eq_init.ramp4h = pd.DataFrame({'ramp4h':[100000]},dtype=float)
    eq_init.ramp3h = pd.DataFrame({'ramp3h':[100000]},dtype=float)
    eq_init.Q_break = conf['para_PSO']['Q_break']
    eq_init.inflow_multiplier = 1
    eq_init.Smin = pd.DataFrame([0],dtype=float,columns=['Smin'])
    eq_init.scenarios = scen

    return eq_init


def create_indicies(nbr_stations, tim, scen, segments, scenario_data):
    # Variables inititalzing with list comprehension
    # Compute the indices (t, i, w) in advance
    hrs = scenario_data.hours.astype(np.int32)
    indices_with_segments = [
        (t,i,w,k)  for t in range(tim)
                for i in range(nbr_stations)
                for w in range(scen)  if hrs[w]>t
                for k in range(segments)]

    indices = [(t, i, w) for t in range(tim) for i in range(nbr_stations) for w in range(scen) if hrs[w]>t]

    indices_only_times = [(t,w) for t in range(tim) for w in range(scen) if hrs[w]>t]
    
    return indices_with_segments, indices, indices_only_times