# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:14:55 2024

@author: yil2
"""

import pandas as pd
import os
import logging
from database_eSett_api import *
from database_entsoe_api import *
from read_process_data import *
from check_missing_data import *
#from datetime import timedelta
import config


def fetch_inflow(country_code):
    ################################Read code informtion for each country####################################

    country_map_filepath='./country_mapping.json'
    logfile_path=config.logfile_hdam
   # Stage 1: Only consider the HDAM type: Inflow=delta(rate)-generation
    country_code_json = pd.read_json(country_map_filepath)

    country_code_json.set_index('Input_code', inplace=True)
    country_code_json = country_code_json[['Entsoe', 'eSett', 'PECD2', 'entose_reservoir_rate', 'entose_generation']]
    country_code_json.index = country_code_json.index.str.strip()

    # Convert specific columns to datetime with UTC localization
    country_code_json[['entose_reservoir_rate', 'entose_generation']] = country_code_json[
        ['entose_reservoir_rate', 'entose_generation']
    ].apply(pd.to_datetime, yearfirst=True, format='ISO8601', errors='coerce', utc=True)

    country_code_list = country_code_json.index

    # Ensure the country code is valid
    assert country_code in country_code_list, f"Warning: unsupported country code: {country_code}"

  
    entsoe_code, eSett_code, PECD2_level, content_start_time, entsoe_generation_time = (
        country_code_json.loc[country_code]
    )

    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            #logging.StreamHandler(),  # Output to console
            logging.FileHandler(logfile_path)  # Output to a text file
        ]
    )


    logger = logging.getLogger(__name__)

    if country_code in ['HR']:
        year=2021
    else:
        year=2018

    #____________________________Reservoir inflow_________________________________
    

    if country_code not in ['SE1', 'SE2', 'SE3', 'SE4','FI']:
        if not (pd.isna(entsoe_generation_time) or pd.isna(content_start_time)) :
            start_date=max(entsoe_generation_time, content_start_time).strftime('%Y%m%d')
            end_date='20220101'  #########max dates from corres
            download_response = {
                'entsoe_dam': True,
                'entsoe_gen': True,
                'esett_gen': False,
            }
        else:
            download_response = {
            'entsoe_dam': False,
            'entsoe_gen': False,
            'esett_gen': False
            }
            print('Warning: There is no reservoir datasets found in entsoe platform')
    else:    #FI and SE should be retrievd from esett
        start_date='20170101'
        end_date='20220101'  #########max dates from corres
        download_response = {
        'entsoe_dam': True,
        'entsoe_gen': False,
        'esett_gen': True
        } 



    #######################retrieve available database###################################
    esett_obj=eSett_response()
    entsoe_obj = entsoe_data_process(api_key=config.entsoe_api_key)

    local_datapath=config.data_dir
    HDAM_gen_response=False
    HDAM_rate_response=False

    if not os.path.isdir(os.path.join(local_datapath, country_code)):
        os.makedirs(os.path.join(local_datapath, country_code))

    # retrieve generation and content
    if download_response['esett_gen']:
        reservoir_generation=esett_obj.eSett_request(eSett_code,country_code)
        HDAM_gen_response=True 
        #TODO: fix it within the api request     
    
    if download_response['entsoe_dam']:
        reservoir_rate=entsoe_obj.entsoe_request(
            "Reservoir rate",entsoe_code,start_date,end_date,country_code)
        HDAM_rate_response=True

    if download_response['entsoe_gen']:
        reservoir_generation=entsoe_obj.entsoe_request(
            "Reservoir generation",entsoe_code,start_date,end_date,country_code)
        HDAM_gen_response=True


        
    #####################process data to get inflow########################################
        
    if HDAM_rate_response and HDAM_gen_response:
        
        
        process=read_process_inflow()

        if country_code in ['FR','ITCS','ITSI']:
            reservoir_generation['New_gen'] = reservoir_generation.fillna(0)[
                ('Hydro Water Reservoir', 'Actual Aggregated')] + reservoir_generation.fillna(0)["Hydro Water Reservoir"]
            
            reservoir_generation=pd.DataFrame(reservoir_generation, columns=['New_gen'])
        elif country_code in ['PT']:
            reservoir_generation['New_gen'] = reservoir_generation[
                ('Hydro Water Reservoir', 'Actual Aggregated')] #negelect the consumption here as they are so small(1-4MWh)
            reservoir_generation=pd.DataFrame(reservoir_generation['New_gen'])
        else:
            reservoir_generation=reservoir_generation
    
        reservoir_generation.columns=['Reservoir generation']
        reservoir_generation=process.index_date(reservoir_generation,'Reservoir generation')
        
    
        reservoir_rate.columns=['Reservoir rate']
        reservoir_rate=process.index_date(reservoir_rate, 'Reservoir rate')
        reservoir_rate.index = reservoir_rate.index.normalize()
        #reservoir_rate.index=reservoir_rate.index-timedelta(hours=1)

        
        
        
        #________________________check the retrieved data:missing or duplicate____________________________
        check_data=Check_fill_data()
        
        start_time= reservoir_generation.index[0]
        end_time=reservoir_generation.index[-1]
        date_range=check_data.create_date_range(start_time, end_time,'h')
        reservoir_generation=check_data.check_duplicate_data(reservoir_generation)
        reservoir_generation=check_data.check_missing_data(reservoir_generation, date_range)  
        reservoir_generation=check_data.check_negative_data(reservoir_generation)
        
        start_time= reservoir_rate.index[0]
        end_time=reservoir_rate.index[-1]
        date_range=check_data.create_date_range(start_time, end_time,'W-SUN')
        reservoir_rate=check_data.check_duplicate_data(reservoir_rate)
        reservoir_rate=check_data.check_missing_data(reservoir_rate, date_range,'W-SUN')    
        reservoir_rate=check_data.check_negative_data(reservoir_rate)   #fill the zero and negative values
        # retrieve price    
        if country_code not in ['ME']:
            
            if country_code in ['HR']:
                price=entsoe_obj.request_price(entsoe_code, '20210101', '20220101',country_code)
            else:
                price=entsoe_obj.request_price(entsoe_code, '20180101', '20190101',country_code)  #entsoe.py has issues on the 'extra day'problem
            
            start_time= price.index[0]
            end_time=price.index[-1]
            date_range=check_data.create_date_range(start_time, end_time,'h')
            price=check_data.check_duplicate_data(price)
            price=check_data.check_missing_data(price, date_range)
            price=check_data.check_negative_data(price)
            
            #____________________save the electricity price___________________
            price_path=os.path.join(config.his_price_path, f"{country_code}_{year}_price.csv")
            
            if not os.path.isdir(config.his_price_path):
                os.makedirs(config.his_price_path)

            if country_code in ['BG']:
                price=price*0.51     #currency transfer is set to be 0.51
            elif country_code in ['RO']:
                price=price*0.2
            
            price=price.drop(price.index[-1])
            price.index=pd.to_datetime(price.index, utc=True)
            if len(price.index)==8760:
                price.columns=['EUR/MWh']
                price.to_csv(price_path,sep=';')
                logger.info(f'Save {year} price for {country_code}--->Finished')
            else: 
                logger.warning('The historical production data is wrong')
        else:
            logger.warning(f'There is no price data for {country_code}')
    
        #_____________________________save the max_min M and P data_________________
        init_params_path=os.path.join(config.his_eq_path, f"{country_code}_initial_params.csv")
        
        init_params= pd.DataFrame({
            'area': [country_code],  
            'max_M(MWh)': [max(reservoir_rate['Reservoir rate']) ],
            'min_M(MWh)': [min(reservoir_rate['Reservoir rate']) ],
            'max_P(MWh)': [max(reservoir_generation['Reservoir generation']) ],
            'min_P(MWh)': [min(reservoir_generation['Reservoir generation']) ]
        })

        logger.info(f'The max and min historical data is: \n,  {init_params}')

        init_params.to_csv(init_params_path, index=False, sep=';')
        logger.info(f'Save the max and min historical data for {country_code}--->Finished')

    
        
        #_____________________save the reservoir generation data for equivalent model_____________________________
        production_path=os.path.join(config.his_eq_path, f"{country_code}_{year}_historical_production.csv")
        
        historical_production=reservoir_generation[reservoir_generation.index.year==year]
        
        if len(historical_production.index)==8760:
            historical_production.to_csv(production_path,sep=';')
            logger.info(f'Save {year} historical production for {country_code}--->Finished')
        else: 
            logger.warning('The historical production data is wrong')
        
    
        
        
        ######################################Inflow calculation############################################
        
        reservoir_generation=process.resample_data(reservoir_generation , 'Reservoir generation','W-SUN')
        reservoir_rate=process.resample_data(reservoir_rate, 'Reservoir rate',  'W-SUN')
        
        
        reservoir_generation, reservoir_rate, inflow_start, inflow_end= process.time_align(reservoir_generation, reservoir_rate)
        inflow_weekly=process.inflow_calculation(reservoir_generation, reservoir_rate)

        inflow_weekly=check_data.check_negative_data(inflow_weekly)

        #____________________save the historical inflow data _____________________________

        inflow_path=os.path.join(config.his_data_path, f"{country_code}_historical_inflow.csv")
        inflow_weekly.to_csv(inflow_path,sep=';')
        logger.info(f'Save historical inflow for {country_code}--->Finished')
        
        #____________________plot and save the historical inflow data _____________________________
        fig_path=os.path.join(config.his_data_path,f'{country_code}_{inflow_start}_{inflow_end}_inflow.pdf')
        process.save_inflow_fig(inflow_weekly, fig_path, country_code)
        
        
    
    else:
        logger.warning('There is no reservoir inflow in this country!')


if __name__ == "__main__":
    country_code = config.country_code 
    fetch_inflow(country_code)




    
    
    


    
