import pandas as pd
import numpy as np
from read_scenario_data import ScenarioData
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ClusterData():
    categories: pd.Series
    length: int
    id: int
    periods_ids: np.array = None
    hours_ids: np.array = None
    periods: int = None
    category: int = None

@dataclass
class TempData():
    price:pd.DataFrame = None
    power:pd.DataFrame = None
    content: pd.DataFrame = None


def periods_single_values(cluster_data: ClusterData, data_name:str, temp_data:TempData,df=None):
    periods = cluster_data.periods
    temp_f = np.zeros(periods) # np.zeros(periods,1)
    countperiods = 0
    for (period, cat) in enumerate(cluster_data.categories):
        if cat == cluster_data.category:
            if data_name == 'hours':
                temp_f[countperiods] = cluster_data.length[period]*24
            elif data_name == 'initialEnergy':
                if temp_data.content:
                    if period == 0:
                        temp_f[countperiods] = temp_data.content.iloc[0][0]
                    else:
                        temp_f[countperiods] = temp_data.content.iloc[cluster_data.id[period-1]*24][0]
                else:
                        temp_f[countperiods] = df

            elif data_name == 'probability':
                temp_f[countperiods] = 1/periods # Equal probability for each scen

            elif data_name == 'scenarios':
                temp_f = np.zeros((1,1))
                temp_f = periods

            elif data_name == 'finalEnergy':
                temp_f[countperiods] = temp_data.content.iloc[cluster_data.id[period]*24-1,0]
            
            elif data_name == 'profit':
                temp_f[countperiods] = (temp_data.price * temp_data.power.values).sum().sum()
            else:
                return -1
            countperiods = countperiods + 1

    return temp_f


def create_periods_ids(cluster_data):

    # Create vector of period IDs (days)
    period_ids = [range(cluster_data.id[0])]
    for i in range(1,len(cluster_data.id)):
        period_ids.append(range(cluster_data.id[i-1], cluster_data.id[i]))
    
    # Find hours corresponding to
     
    # hour_ids = np.empty[range(cluster_data.id[1]*24)]  
    hour_ids = np.empty((len(cluster_data.id)),dtype=object)
    for i in range(len(hour_ids)):
        if i == 0:
            hour_ids[i] = range(cluster_data.id[i]*24)
        else:
            hour_ids[i]=(range(cluster_data.id[i-1]*24, cluster_data.id[i]*24))

    return period_ids, hour_ids


def peridos_hourly_data(df: pd.DataFrame, cluster_data: ClusterData):
    if not isinstance(df,pd.DataFrame):
        df = pd.DataFrame(df)
    partitioned_data = pd.DataFrame(
        {f'scen{i}': [] for i in range(1, cluster_data.periods + 1)}, dtype=float)
    countperiods = 0 # For assigning data to correct column of temp_m
    periodnumber = np.zeros((len(cluster_data.id)), dtype=int) # Track the order of periods in category
    whichperiod = np.zeros((len(cluster_data.id))) # Store which periods are in category
    temp_m = np.empty((df.shape[0], cluster_data.periods),dtype=object) # Store data
    hour_ids = cluster_data.hours_ids

    # Extract data from dataframe into a matrix
    for period, cat in enumerate(cluster_data.categories):
        if cat == cluster_data.category:
            periodnumber[period] = countperiods
            whichperiod[period] = 1
            temp_m[hour_ids[period], countperiods] = np.array(df.loc[hour_ids[period], :]).T[0]
            countperiods += 1

    # Extract vectors from temp_m
    temp_vectors = [None] * cluster_data.periods
    for check, period in enumerate(whichperiod):
        if whichperiod[check] == 1:
            temp_vectors[periodnumber[check]] = temp_m[hour_ids[check], periodnumber[check]]

    # Find the length of the longest vector
    max_length = max(len(vec) for vec in temp_vectors)
    
    # Pad the shorter vectors with zeros
    padded_vectors = np.column_stack(
        [np.concatenate((vec, np.nan*np.zeros(max_length - len(vec)))) for vec in temp_vectors])
    # Push padded data into empty dataframe
    padded_vectors = pd.DataFrame(padded_vectors, columns=partitioned_data.columns)
    partitioned_data = pd.concat([partitioned_data, padded_vectors])

    return partitioned_data


def periods_m0_share(df, cluster_data:ClusterData):
    partitioned_data = pd.DataFrame(
        {f'scen{i}': [] for i in range(1, cluster_data.periods + 1)}, dtype=float)
    countperiods = 0 # For assigning data to correct column of temp_m
    temp_m = np.zeros((df.shape[0], cluster_data.periods)) # Store data
    # Extract data from dataframe into a matrix
    for (period, cat) in enumerate(cluster_data.categories):
        if cat == cluster_data.category:
            temp_m[:,countperiods] = df.loc[:,df.columns[period]]
            countperiods = countperiods + 1
    # Push data into empty dataframe
    temp_m = pd.DataFrame(temp_m, columns=partitioned_data.columns)
    partitioned_data = pd.concat([partitioned_data, temp_m])

    return partitioned_data


def periods_min_share(df:pd.Series, cluster_data:ClusterData):
    # MTminShare TODO: could this not be the same?
    partitioned_data = pd.DataFrame(
        {f'scen{i}': [] for i in range(1, cluster_data.periods + 1)}, dtype=float)
    countperiods = 0 # For assigning data to correct column of temp_m
    temp_m = np.zeros(cluster_data.periods) # Store data    
    # Extract data from dataframe into a matrix
    for (period, cat) in enumerate(cluster_data.categories):
        if cat == cluster_data.category:
            temp_m[countperiods] = df[df.index[period]]
            countperiods = countperiods + 1
    # Push data into empty dataframe
    temp_m = pd.DataFrame(temp_m.reshape(1,-1), columns=partitioned_data.columns)
    partitioned_data = pd.concat([partitioned_data, temp_m])

    return partitioned_data


def periods_dates(df:pd.DataFrame, cluster_data:ClusterData):
    partitioned_data = pd.DataFrame(
        {f'scen{i}': [] for i in range(1, cluster_data.periods + 1)}, dtype=float)
    
    countperiods = 0 # For assigning data to correct column of temp_m
    periodnumber = np.zeros((len(cluster_data.id)), dtype=int) # Track the order of periods in category
    whichperiod = np.zeros((len(cluster_data.id))) # Store which periods are in category
    temp_m = np.empty((df.shape[0], cluster_data.periods),dtype=object) # Store data
    period_ids = cluster_data.periods_ids

    # Extract data from dataframe into a matrix
    for (period, cat) in enumerate(cluster_data.categories):
        if cat == cluster_data.category:
            periodnumber[period] = countperiods
            whichperiod[period] = 1
            temp_m[period_ids[period],countperiods] = df.loc[period_ids[period]].T.values[0]
            countperiods = countperiods + 1
            
    # Extract vectors from temp_m
    temp_vectors = [None] * cluster_data.periods
    for check, period in enumerate(whichperiod):
        if whichperiod[check] == 1:
            temp_vectors[periodnumber[check]] = temp_m[period_ids[check], periodnumber[check]]

    # Find the length of the longest vector
    max_length = max(len(vec) for vec in temp_vectors)
    # Pad the shorter vectors with zeros
    padded_vectors = np.column_stack([np.concatenate((vec, np.zeros(max_length - len(vec)))) for vec in temp_vectors])

    padded_vectors[padded_vectors==0] = '2222-01-01' , # Set any date for unused values

    # Push padded data into empty dataframe
    padded_vectors = pd.DataFrame(padded_vectors, columns=partitioned_data.columns)
    partitioned_data = pd.concat([partitioned_data, padded_vectors])
    # Edit data in structs to fit the requested category
    return partitioned_data


def periods_arrays(df, data,data_name, cluster_data):
    if df.shape[0] > cluster_data.id.iloc[-1]*24-1: # df is Hourly data
        temp_f = peridos_hourly_data(df, cluster_data)

    elif (data_name == 'M0share') & (isinstance(data,ScenarioData)): # df is M-values
        temp_f = periods_m0_share(df, cluster_data=cluster_data)

    elif (data_name == 'MTminShare') & (isinstance(data,ScenarioData)) : # MTminShare
        temp_f = periods_min_share(df,cluster_data=cluster_data)

    elif isinstance(data,ScenarioData) : # df is Dates (or ID = 1)
        temp_f = periods_dates(df=df,cluster_data=cluster_data)

    return temp_f


def initalize_data(data, temp_data, cluster_data:ClusterData):
    
    # Count the number of periods in given category
    cluster_data.periods = sum(cluster_data.categories[i] == cluster_data.category for i in range(
        len(cluster_data.categories)))
    logger.info((f"Periods in category: {cluster_data.periods}"))

    cluster_data.periods_ids, cluster_data.hours_ids = create_periods_ids(cluster_data)
   
    data_dict = vars(data)
    
    for data_name, df  in data_dict.items():
        if data_name == 'thermal':
            continue
        elif data_name =='thermal_power':
            continue
        if df is None:
            logger.info((f"Skipping {data_name} to create period data"))
            continue    

        if isinstance(df, (int,np.int64)) or isinstance(df, float): # Data is a float/int
            temp_f = periods_single_values(cluster_data, data_name, temp_data,df)
            if not isinstance(temp_f,np.ndarray): # Some varaibles do not need to split into periods
                continue
            setattr(data, data_name, temp_f)
        elif (len(df)==1) & (data_name != 'MTminShare'):
            temp_f = periods_single_values(cluster_data, data_name, temp_data)
            setattr(data, data_name, temp_f)
        else:
            temp_f = periods_arrays(df, data,data_name, cluster_data)
            setattr(data, data_name, temp_f)

    return data


def read_cluster_data(conf, train_scenario):
    category = conf['para_general']['category']
    
    if 'cluster' in conf['file_location']:
        cluster_file = conf['file_location']['cluster']
        clustering_df = pd.read_csv(cluster_file, delimiter= ';',header=None)
        cluster_data = ClusterData(
            categories=clustering_df.loc[:,0],
            length=clustering_df.loc[:,1],
            id=clustering_df.loc[:,2],
            category=category)
    else:
        # Inilitlize cluster data  when only running one category
        cluster_data = ClusterData(
            categories=pd.Series([1]),
            length = pd.Series([len(train_scenario.days)]),
            id = pd.Series([len(train_scenario.days)]),
            category=category)

    return cluster_data


def create_power_production_curve(original_system):
    stacked_orig_power = original_system.power.stack().reset_index(drop=True)
    original_system.power_production_curve = stacked_orig_power.sort_values(
        ascending=False).reset_index(drop=True)

    return original_system

def get_period_data(
        train_scenario, conf):
    
    cluster_data = read_cluster_data(conf, train_scenario)

    temp_data = TempData(
        price=train_scenario.price)

    # initalize data: 
    train_scenario = initalize_data(
        train_scenario,
        temp_data,
        cluster_data)


    return train_scenario