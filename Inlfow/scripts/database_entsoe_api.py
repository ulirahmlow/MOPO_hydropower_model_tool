# entsoe data request API package 
from entsoe import EntsoePandasClient
#from entsoe import EntsoeRawClient
from pandas import Timestamp,read_csv
from pathlib import Path
import pandas as pd
import config
import logging

###IMPROVEMENT: create the folder







class entsoe_data_process:

    def __init__(self, **kwargs):
        self.api_key = kwargs['api_key']
        self.client = EntsoePandasClient(api_key=self.api_key)
        # entsoe API parameters
        self.__prstype = {"hydro_pumped_storage"     : "B10",
                          "hydro_river_and_poundage" : "B11",
                          "hydro_water_reservoir"    : "B12"}
        
        self.__area_code = ['DE_50HZ', 'AL', 'DE_AMPRION', 'AT', 'BY', 'BE', 'BA', 'BG', 
                            'CZ_DE_SK', 'HR', 'CWE', 'CY', 'CZ', 'DE_AT_LU', 'DE_LU',
                            'DK', 'DK_1', 'DK_1_NO_1', 'DK_2', 'DK_CA', 'EE', 'FI', 'MK', 
                            'FR', 'DE', 'GR', 'HU', 'IS', 'IE_SEM', 'IE', 'IT', 'IT_SACO_AC', 
                            'IT_CALA', 'IT_SACO_DC', 'IT_BRNN', 'IT_CNOR', 'IT_CSUD', 'IT_FOGN',
                            'IT_GR', 'IT_MACRO_NORTH', 'IT_MACRO_SOUTH', 'IT_MALTA', 'IT_NORD',
                            'IT_NORD_AT', 'IT_NORD_CH', 'IT_NORD_FR', 'IT_NORD_SI', 'IT_PRGP',
                            'IT_ROSN', 'IT_SARD', 'IT_SICI', 'IT_SUD', 'RU_KGD', 'LV', 'LT',
                            'LU', 'LU_BZN', 'MT', 'ME', 'GB', 'GB_IFA', 'GB_IFA2', 'GB_ELECLINK', 
                            'UK', 'NL', 'NO_1', 'NO_1A', 'NO_2', 'NO_2_NSL', 'NO_2A', 'NO_3',
                            'NO_4', 'NO_5', 'NO', 'PL_CZ', 'PL', 'PT', 'MD', 'RO', 'RU', 'SE_1', 
                            'SE_2', 'SE_3', 'SE_4', 'RS', 'SK', 'SI', 'GB_NIR', 'ES', 'SE', 'CH', 
                            'DE_TENNET', 'DE_TRANSNET', 'TR', 'UA', 'UA_DOBTPP', 'UA_BEI', 'UA_IPS', 
                            'XK', 'DE_AMP_LU']
        
        self.__local_data_path = config.data_dir

        self.__time_zone = "UTC"
        self.logger = logging.getLogger(__name__)
        #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # request_data 
    def request_data(self, data_type, area, start_date, end_date):

        assert area in self.__area_code, "unsupported area code"
        start = Timestamp(start_date, tz=self.__time_zone)
        end = Timestamp(end_date, tz=self.__time_zone)

        if data_type == "Reservoir rate":
            # Pandas Series
            return self.client.query_aggregate_water_reservoirs_and_hydro_storage(area, start=start, end=end)
        elif data_type == "Reservoir generation":
            # Pandas Dataframe 1 row
            return self.client.query_generation(area, start=start, end=end,psr_type=self.__prstype["hydro_water_reservoir"])
        elif data_type == "Pumped Storage":
            # Pandas Dataframe 1 row
            return self.client.query_generation(area, start=start, end=end,psr_type=self.__prstype["hydro_pumped_storage"])
        elif data_type == "Run of river":
            # Pandas Dataframe 1 row
            return self.client.query_generation(area, start=start, end=end,psr_type=self.__prstype["hydro_river_and_poundage"])
        else:
            raise ValueError("Unsupported data request.")
                 
        
    # update local data
    def entsoe_request(self, data_type: str, area: str, start_date: str, end_date: str,country_code:str)->pd.DataFrame:

        assert area in self.__area_code, "unsupported area code"
        start = Timestamp(start_date, tz=self.__time_zone)
        end = Timestamp(end_date, tz=self.__time_zone)


        if data_type == "Reservoir rate":
            # Pandas Series
            requested_data = self.client.query_aggregate_water_reservoirs_and_hydro_storage(area, start=start, end=end)
            data_path = Path(__file__).parent / self.__local_data_path / country_code/ f"{country_code}_{start_date}_20211231_reservoir rate.csv"
        elif data_type == "Reservoir generation":
            # Pandas Dataframe 1 row
            requested_data = self.client.query_generation(area, start=start, end=end,psr_type=self.__prstype["hydro_water_reservoir"])
            data_path = Path(__file__).parent / self.__local_data_path / country_code / f"{country_code}_{start_date}_20211231_reservoir generation.csv"
        elif data_type == "Pumped Storage":
            # Pandas Dataframe 1 row
            requested_data = self.client.query_generation(area, start=start, end=end,psr_type=self.__prstype["hydro_pumped_storage"])
            data_path = Path(__file__).parent / self.__local_data_path/ country_code  / f"{country_code}_{start_date}_20211231_pump_generation.csv"
        elif data_type == "Run of river":
            # Pandas Dataframe 1 row
            requested_data = self.client.query_generation(area, start=start, end=end,psr_type=self.__prstype["hydro_river_and_poundage"])
            data_path = Path(__file__).parent / self.__local_data_path/ country_code  / f"{country_code}_{start_date}_20211231_ror_generation.csv"
        else:
            raise ValueError("Unsupported data request.")
        requested_data.to_csv(str(data_path))
        
        self.logger.info(f"Retrieve entsoe data: {country_code}_{data_type}--->Finished")
        return pd.DataFrame(requested_data)

    # local_data_list 
    def local_data_list (self):
        file_path_list = []
        for file_path in Path(__file__).parent.joinpath(self.__local_data_path).glob("*.csv"):
            file_path_list.append(str(file_path))
            print(file_path.name)
        return file_path_list
    
    # read_local_data
    def read_local_data (self, file_path):
        return read_csv(file_path)
    
    
    def request_price(self, area, start_date, end_date,country_code):
        #client = EntsoeRawClient(api_key=self.api_key)
        assert area in self.__area_code, "unsupported area code"
        
        start = Timestamp(start_date, tz=self.__time_zone)
        end = Timestamp(end_date, tz=self.__time_zone)   
        request_price=self.client.query_day_ahead_prices(area, start=start, end=end)
        
        self.logger.info(f"Retrieve entsoe data: {country_code}_price--->Finished")
        
        return pd.DataFrame(request_price)
