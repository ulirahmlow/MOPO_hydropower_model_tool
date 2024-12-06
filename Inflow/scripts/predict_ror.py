import pandas as pd
#import datetime as date
from predict_inflow import predictor
import os
#import pathlib as Path
from read_process_data import *
import logging
#mport glob
import config



def ror_P_prediction(country_code):
    process=read_process_inflow()
    if country_code in ['AT']:
        predicted_year=2021
    elif country_code in ['CH','CZ']:
        predicted_year=2019
    else:
        predicted_year=2020
    predict=predictor(country_code, predicted_year)

    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler(config.logfile_pred_hror)  # Output to a text file
        ]
    )


    logger = logging.getLogger(__name__)

    weekly=False

    sf_runoff_filename='Surface_runoff_micrometers.csv'
    sub_runoff_filename='Sub_surface_runoff_micrometers.csv'
    weather_path=config.corres_data_path

    sf_runoff_filename='Surface_runoff_micrometers.csv'
    sub_runoff_filename='Sub_surface_runoff_micrometers.csv'

    sf_runoff_path=os.path.join(config.corres_data_path,  sf_runoff_filename)
    sub_runoff_path=os.path.join(config.corres_data_path,  sub_runoff_filename)

    surface_runoff=pd.read_csv(sf_runoff_path, index_col='time')
    sub_runoff=pd.read_csv(sub_runoff_path, index_col='time')    

    inflow_path=os.path.join(config.his_data_path,f'{country_code}_historical_ror_generation.csv')
    # for csv in glob.glob(ror_path):
    #     inflow_path=csv

    #inflow_path=f'../data/{country_code}/{country_code}_ror_generation.csv'
    inflow=pd.read_csv(inflow_path, sep=';',index_col='Unnamed: 0')

    inflow.index=pd.to_datetime(inflow.index ,utc=True, errors='coerce')
    start_time, end_time=inflow.index[0], inflow.index[-1]
    surface_runoff.index=pd.to_datetime(surface_runoff.index, utc=True, errors='coerce')
    sub_runoff.index=pd.to_datetime(sub_runoff.index,  utc=True, errors='coerce')

    if weekly==True:
        # ____________________________________________weekly regression_________________________________________________________
        input_X=pd.concat([sub_runoff[sub_runoff.index>=start_time], surface_runoff[surface_runoff.index>=start_time]],axis=1)
        X, y = predict.data_preprocess(input_X, inflow, frequency='W-SUN', aggregation=False) 


        image_path=config.figs_path_hror
        y_pred,y_test,regr_model=predict.regression(X, y,image_path, positive=True)
        predict.evaluation(y_pred, y_test)
        inflow_pred=predict.desample(y_pred, 'hourly')
        inflow_test=predict.desample(y_test, 'hourly')


        predict.save_fig(inflow_pred, inflow_test, 7, image_path)

        #___________________predict the long-term inflow_____________________

        weather_long=pd.concat([sub_runoff, surface_runoff],axis=1)
        weather_long=weather_long.resample('W-SUN').sum()
        inflow_longterm=regr_model.predict(weather_long)
        inflow_longterm=pd.DataFrame(inflow_longterm)
        inflow_longterm.set_index(weather_long.index, inplace=True)
        filename=f"{country_code}_1991_2021_weekly_ror_P_mwh.csv"
        predict.save_file(inflow_longterm,filename)

        inflow_longterm_hourly=inflow_longterm.resample('h').ffill().div(168)
        predict.save_fig_longterm(inflow_longterm_hourly, image_path)
        filename=f"{country_code}_1991_2021_hourly_ror_P_mwh.csv"
        predict.save_file(inflow_longterm_hourly,filename)

    else:
        #____________________________________________daily regression with lags_______________________________________________
        input_X=pd.concat([sub_runoff[sub_runoff.index>=start_time], surface_runoff[surface_runoff.index>=start_time]],axis=1)
        input_X=input_X.resample('d').sum()
        input_X = predict.create_lags(input_X, lags=60)
        X, y = predict.data_preprocess(input_X, inflow, frequency='d', aggregation=False) 

        image_path=config.figs_path_hror
        y_pred,y_test,regr_model=predict.regression(X, y,image_path, positive=True)
        predict.evaluation(y_pred, y_test)
        inflow_pred=predict.desample(y_pred, 'daily')
        inflow_test=predict.desample(y_test, 'daily')

        predict.save_fig(inflow_pred, inflow_test, 24, image_path)

        #___________________predict the long-term inflow_____________________

        weather_long=pd.concat([sub_runoff, surface_runoff],axis=1)
        weather_long=weather_long.resample('d').sum()
        weather_long = predict.create_lags(weather_long, lags=60)
        inflow_longterm=regr_model.predict(weather_long)
        inflow_longterm=pd.DataFrame(inflow_longterm)
        inflow_longterm.set_index(weather_long.index, inplace=True)
        filename=f"{country_code}_1991_2021_daily_ror_P_mwh.csv"
        predict.save_file(inflow_longterm,filename)

        inflow_longterm_hourly=inflow_longterm.resample('h').ffill().div(24)
        predict.save_fig_longterm(inflow_longterm_hourly, image_path)
        filename=f"{country_code}_1991_2021_hourly_ror_P_mwh.csv"
        predict.save_file(inflow_longterm_hourly,filename)


if __name__ == '__main__':
    country_code = config.country_code
    ror_P_prediction(country_code)