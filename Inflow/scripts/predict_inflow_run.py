# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:57:38 2024

@author: yil2
"""

import pandas as pd
from predict_inflow import predictor
from read_process_data import *
import logging
import config
import os

def inflow_prediction(country_code):
    if country_code in ['HR']:
        predicted_year=2021
    else:
        predicted_year=2018
    predict=predictor(country_code, predicted_year)

    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler(config.logfile_pred_hdam)  # Output to a text file
        ]
    )


    logger = logging.getLogger(__name__)



    sf_runoff_filename='Surface_runoff_micrometers.csv'
    sub_runoff_filename='Sub_surface_runoff_micrometers.csv'

    sf_runoff_path=os.path.join(config.corres_data_path,  sf_runoff_filename)
    sub_runoff_path=os.path.join(config.corres_data_path,  sub_runoff_filename)

    #_______________________run it once for FR,ES case 
    if country_code in ['FR','ES']:
        if not os.isdir(config.corres_data_path):
            weather_path=os.path.join(config.weather_dir, "{country_code}1\\results\\")
            surface_runoff_first=pd.read_csv(weather_path+sf_runoff_filename, index_col='time')
            sub_runoff_first=pd.read_csv(weather_path+sub_runoff_filename, index_col='time')
            
            weather_path=os.path.join(config.weather_dir, "{country_code}2\\results\\")
            surface_runoff_second=pd.read_csv(weather_path+sf_runoff_filename, index_col='time')
            sub_runoff_second=pd.read_csv(weather_path+sub_runoff_filename, index_col='time')
            
            
            surface_runoff= pd.concat([surface_runoff_first,surface_runoff_second], axis=1)
            sub_runoff=pd.concat([sub_runoff_first, sub_runoff_second], axis=1)
            os.makedirs(config.corres_data_path)
            surface_runoff.to_csv(sf_runoff_path)
            sub_runoff.to_csv(sf_runoff_path)
        

    surface_runoff=pd.read_csv(sf_runoff_path, index_col='time')
    sub_runoff=pd.read_csv(sub_runoff_path, index_col='time')   
    inflow_path=os.path.join(config.his_data_path, f'{country_code}_historical_inflow.csv')

    inflow=pd.read_csv(inflow_path, sep=';',index_col='Unnamed: 0')

    inflow.index=pd.to_datetime(inflow.index ,utc=True, errors='coerce')
    start_time, end_time=inflow.index[0], inflow.index[-1]
    surface_runoff.index=pd.to_datetime(surface_runoff.index, utc=True, errors='coerce')
    sub_runoff.index=pd.to_datetime(sub_runoff.index,  utc=True, errors='coerce')




    input_X=pd.concat([sub_runoff[sub_runoff.index>=start_time], surface_runoff[surface_runoff.index>=start_time]],axis=1)
    X, y = predict.data_preprocess(input_X, inflow, frequency='W-SUN', aggregation=False) 

    image_path=config.figs_path_hdam
    y_pred,y_test,regr_model=predict.regression(X, y,image_path, positive=True)
    predict.evaluation(y_pred, y_test)
    inflow_pred=predict.desample(y_pred, 'hourly')
    inflow_test=predict.desample(y_test, 'hourly')

    predict.save_fig(inflow_pred, inflow_test, 7, image_path)

    #___________________predict the long-term inflow_____________________

    weather_long=pd.concat([sub_runoff, surface_runoff],axis=1)
    weather_long=weather_long.resample('W-Mon').sum()
    inflow_longterm=regr_model.predict(weather_long)
    inflow_longterm=pd.DataFrame(inflow_longterm)
    inflow_longterm.set_index(weather_long.index, inplace=True)
    filename=f"{country_code}_1991_2021_weekly_inflow_mwh.csv"
    predict.save_file(inflow_longterm,filename)


    inflow_longterm_hourly=inflow_longterm.resample('h').ffill().div(168)
    predict.save_fig_longterm(inflow_longterm_hourly, image_path)
    filename=f"{country_code}_1991_2021_hourly_inflow_mwh.csv"
    predict.save_file(inflow_longterm_hourly,filename)
    #____________________save 2018 data____________________________
    inflow_eq=inflow_longterm_hourly[inflow_longterm_hourly.index.year==predicted_year]
    file_name=f"{country_code}_{predicted_year}_hourly_inflow.csv"
    predict.save_file(inflow_eq,file_name)



    # coef.columns=list(runoff.columns) + list(runoff.columns)

    # const=[0]
    # const=pd.DataFrame(const)
    # const.columns=['const']

    # regr=pd.concat([coef,const],axis=1)
    # solution_path='../solutions/linear_regression/'
    #regr.to_csv(solution_path+solutionfile)


if __name__ == "__main__":
    country_code = config.country_code
    inflow_prediction(country_code)



    
    







