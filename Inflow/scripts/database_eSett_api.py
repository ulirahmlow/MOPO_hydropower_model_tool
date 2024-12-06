# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:53:42 2024

@author: yil2
"""

import requests
import pandas as pd
from pathlib import Path
import logging
import config

class eSett_response:
    
    def __init__(self):
        
        self.url="https://api.opendata.esett.com/EXP16/MBAOptions"
        self.__local_data_path =  config.data_dir
        self.__MBA_code=['SE1', 'SE2', 'SE3', 'SE4','FI']
        self.logger = logging.getLogger(__name__)
        #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_mba_info(self, MBA):
        self.response = requests.get(self.url)    #check the mba code in the string of api request
        #print(response.status_code) # 200: Everything went okay, and the result has been returned (if any).
        self.mba_codelist=self.response.json()
        
        for country in self.mba_codelist:
            for mba in country["mbas"]:
                if mba["name"] == MBA:
                    return mba["code"]
        return None
    
    def eSett_request(self, area,country_code):
        assert area in self.__MBA_code, "unsupported area code"
        mba_code= self.get_mba_info(area)
        if mba_code:
            # Build the API URL dynamically
            api_url = f"https://api.opendata.esett.com/EXP16/Aggregate?end=2024-06-30T00%3A00%3A00.000Z&mba={mba_code}&mba=string&resolution=hour&start=2017-05-01T00%3A00%3A00.000Z"
            # Make the API request
            data_response = requests.get(api_url)
            data=data_response.json()
            ####store data into csv file
            data=pd.DataFrame(data)
            hydro=data[['timestamp','hydro']]
            #hydro.columns=[['Date CET/CEST','hydro']]
            file_path= Path(__file__).parent / self.__local_data_path / country_code/ f"{country_code}_reservoir generation.csv"
            hydro.to_csv(file_path, index=False)
            hydro=hydro.set_index(hydro['timestamp'], inplace=False, drop=True)
            hydro=hydro.drop(columns='timestamp')

            self.logger.info(f"Retrieve eSett data: {country_code}_generation--->Finished")
            
        else:
            self.logger.warn(f"MBA '{area}' not found.")   
            file_path=None
            
        return pd.DataFrame(hydro)
            
        
