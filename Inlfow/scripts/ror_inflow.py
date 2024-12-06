# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:22:57 2024

@author: yil2
"""

import pandas as pd
import geopandas as gpd
import os
import logging
import matplotlib.pyplot as plt
from database_entsoe_api import *
from read_process_data import *
from check_missing_data import *

def fetch_ror(country_code):
    ################################Read code informtion for each country####################################

    country_map_filepath='./country_mapping.json'
    # consider the HDAM type: Inflow=delta(rate)-generation
    # Load JSON file
    country_code_json = pd.read_json(country_map_filepath)
    country_code_json.set_index('Input_code', inplace=True)
    country_code_json = country_code_json[['Entsoe', 'PECD2', 'entose_ror']]
    country_code_json.index = country_code_json.index.str.strip()

    # Convert 'entose_ror' to datetime with UTC localization
    country_code_json['entose_ror'] = pd.to_datetime(
        country_code_json['entose_ror'], yearfirst=True, format='ISO8601'
    ).dt.tz_localize('UTC')

    country_code_list = country_code_json.index


    assert country_code in country_code_list, f"unsupported country code : {country_code}"
    entsoe_code, PECD2_level, ror_start_time = country_code_json.loc[country_code]

    logfile_path=config.logfile_hror
    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            #logging.StreamHandler(),  # Output to console
            logging.FileHandler(logfile_path)  # Output to a text file
        ]
    )


    logger = logging.getLogger(__name__)

    entsoe_obj = entsoe_data_process(api_key = config.entsoe_api_key)

    local_datapath=config.data_dir

    image_path=os.path.join(local_datapath,f"{country_code}\\{country_code}_historical_ror_inflow.pdf")
    if not os.path.isdir(os.path.join(local_datapath, country_code)):
        os.makedirs(os.path.join(local_datapath, country_code))

    if  pd.notna(ror_start_time):
        end_date='20211231' 
        if country_code in ['ITSU','LT']:
            end_date='20231231'
        start_date=ror_start_time.strftime('%Y%m%d')
        ror=entsoe_obj.entsoe_request("Run of river", entsoe_code,start_date,end_date,country_code)
        if country_code in ['AT','DK','IE','PT']:
            #dam_gen=entsoe_obj.entsoe_request( "Run of river", entsoe_code,start_date,end_date,country_code)
            ror['ror']=ror.fillna(0)[('Hydro Run-of-river and poundage',  'Actual Aggregated')]+ror.fillna(0)[('Hydro Run-of-river and poundage',  'Actual Consumption')]
            ror=ror['ror']
            
            ror=pd.DataFrame(ror)
        elif country_code in ['FI','FR','ITSA','SI']:
            ror['ror']=ror.fillna(0)[('Hydro Run-of-river and poundage',  'Actual Aggregated')]+ror.fillna(0)[
                'Hydro Run-of-river and poundage']
            ror=pd.DataFrame(ror['ror'])

        elif country_code in ['ITCS']:   #ITSI has connection error of entsoe, but litte reservoir generation, negelect
            dam_gen=entsoe_obj.entsoe_request( "Reservoir generation", entsoe_code,start_date,end_date,country_code)
            #dam_gen['New_gen'] = dam_gen.fillna(0)[
            #    ('Hydro Water Reservoir', 'Actual Aggregated')] + dam_gen.fillna(0)["Hydro Water Reservoir"]
            ror['ror']=ror.fillna(0)['Hydro Run-of-river and poundage']+dam_gen.fillna(0)[
                ('Hydro Water Reservoir',  'Actual Aggregated')]+ dam_gen.fillna(0)["Hydro Water Reservoir"]
            ror=pd.DataFrame(ror['ror'])
            #dam_gen=pd.DataFrame(dam_gen['New_gen'])
            
        else:
            ror=pd.DataFrame(ror)
        
        ror.columns=['Run of River Generation']
        process=read_process_inflow()
        ror=process.index_date(ror, 'Run of River Generation')

        # if country_code in ['BA']:
        #     ror.loc['2021-12-07 16:00:00+00:00']= ror.loc['2021-12-07 15:00:00+00:00']

    #_______________________check data________________________________________________
        check_data=Check_fill_data()
        
        start_time= ror.index[0]
        end_time=ror.index[-1]
        date_range=check_data.create_date_range(start_time, end_time,'h')
        ror=check_data.check_duplicate_data(ror)
        ror=check_data.check_missing_data(ror, date_range)  
        ror=check_data.check_negative_data(ror)
        ror_path=os.path.join(local_datapath,f'{country_code}\\{country_code}_historical_ror_generation.csv')
        #ror=process.filldata(ror, country_code, ror_path)
        ror.to_csv(ror_path,sep=';')
        logger.info(f'Save historical ror generation for {country_code}--->Finished')

        ror.plot(figsize=(12,5), label='Run of River Weekly')
        plt.title(f"ROR generation in {country_code}")
        plt.xlabel('Time')
        plt.ylabel('ROR generation (MWh)')
        plt.legend()
        plt.savefig (image_path, bbox_inches='tight')
        plt.show()
        logger.info(f'Save historical ror fig for {country_code}--->Finished')
        
    else:
        print('Warning: There is no ror inflow in this country!')


if __name__ == '__main__':
    country_code = config.country_code
    fetch_ror(country_code)