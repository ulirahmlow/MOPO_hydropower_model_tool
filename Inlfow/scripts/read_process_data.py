# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:05:00 2024

@author: yil2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import read_csv
import logging
import config

class read_process_inflow:
    def __init__(self):
       self.__local_data_path =  config.data_dir
       self.logger = logging.getLogger(__name__)
       #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 
        
    def read_local_data (self, file_path):
        return read_csv(file_path)
    
    
    def index_date(self, input_data:pd.DataFrame, value_column:str)->pd.DataFrame:
        
        """
        Preprocesses the given data into a datetime-indexed DataFrame

        Parameters
        ----------
        input_data : pd.DataFrame
            The input data to be preprocessed.
        datetime_column : str
            The column name to be converted to datetime and used as the index.
        value_column : str
            The column name of the value data.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame with datetime index and value column.
        """
        input_data[value_column] = pd.to_numeric(input_data[value_column], errors='coerce')
        input_data.index = pd.to_datetime(input_data.index, utc=True, errors='coerce')
        input_data.dropna(inplace=True)
        #input_data.set_index(datetime_column, inplace=True)
        return pd.DataFrame(input_data)
        
    def resample_data(self, input_data:pd.DataFrame, value_column:str, freq='W-SUN')->pd.DataFrame:
        
        """
        Resamples the given data into a DataFrame with a given frequency.

        Parameters
        ----------
        input_data : pd.DataFrame
            The input data to be resampled.
        value_column : str
            The column name of the value data.
        freq : str, optional
            The frequency of the resampled data. The default is 'W-SUN', which means weekly data
            with the week starting on Sunday.

        Returns
        -------
        pd.DataFrame
            The resampled DataFrame with datetime index and value column.
        """
        input_data[value_column] = pd.to_numeric(input_data[value_column], errors='coerce')
        input_data=input_data.resample(freq).sum()
        
        return pd.DataFrame(input_data)
    
    def pre_process(self, sampledata,datatime_column, value_column):
        sampledata=sampledata.dropna(how='any')
        sampledata[value_column]=sampledata[value_column].astype(float)
        sampledata[datatime_column] = pd.to_datetime(sampledata[datatime_column],utc=True)
                                                                
        #sampledata[datatime_column] = sampledata[datatime_column].dt.normalize()  
        #sampledata[datatime_column]=sampledata[datatime_column].dt.date                                           
                                                                
        sampledata.set_index(datatime_column, inplace=True)
        sampledata.index = pd.to_datetime(sampledata.index, utc=True)
        
        return sampledata
        


    def time_align(self, generation:pd.DataFrame, content:pd.DataFrame)->tuple:
       
        #find the start time of two dataframe
        """
        Aligns the time index of two DataFrames (generation and content)
        by finding the common time range.

        Parameters
        ----------
        generation : pd.DataFrame
            The generation data.
        content : pd.DataFrame
            The content data.

        Returns
        -------
        tuple
            A tuple containing the aligned generation and content DataFrames, and the start and end years of the aligned data.
        """
        
        start_time_both = max(generation.index[0], content.index[0])
    
        #find the end time
        end_time_both = min(generation.index[-1], content.index[-1])
    
        # slice the data within (start time, end time)
        generation=generation[start_time_both:end_time_both]
        content=content[start_time_both:end_time_both]
    
        generation=pd.DataFrame(generation)
        content=pd.DataFrame(content)
    
        return generation,content,generation.index[0].year,generation.index[-1].year
    
    
    
    
    def inflow_calculation(self, generation:pd.DataFrame, content:pd.DataFrame)->pd.DataFrame:
        """
        Calculates the inflow data based on the given generation and content data.

        Parameters
        ----------
        generation : pd.DataFrame
            The generation data.
        content : pd.DataFrame
            The content data.

        Returns
        -------
        pd.DataFrame
            The calculated inflow data.
        """

        #content = content.apply(pd.to_numeric, errors='coerce')
        #generation = generation.apply(pd.to_numeric, errors='coerce')
        inflow=pd.DataFrame(index=content.index[1:], columns=['Inflow weekly']) 
        for t in range(1, len(generation)):
            inflow.loc[content.index[t]]=content.iloc[t,0]-content.iloc[t-1,0]+generation.iloc[t,0] 
        inflow['Inflow weekly']=pd.to_numeric(inflow['Inflow weekly'], errors='coerce')
        
        return pd.DataFrame(inflow)
    
    def filldata(self, data, area, file_path):
        data=data.mask(data<0)
        pd.set_option('future.no_silent_downcasting', True)
        # data_ffilled=data.fillna(method='ffill')
        # data_bfilled=data.fillna(method='bfill')
        data_ffilled = data.ffill().infer_objects(copy=False)
        data_bfilled = data.bfill().infer_objects(copy=False)
        mean_filled_value=(data_bfilled+data_ffilled)/2
        data.fillna(mean_filled_value,inplace=True)
        data.to_csv(str(file_path), sep=';')  #save by sep=; for Uli
                    
        self.logger.info('Inflow calculation --> Finished')
        
        return data
    
            
    
            
    def save_inflow_fig(self, data:pd.DataFrame, path:str, country_code:str)->None:

        """
        Saves a plot of the inflow data for a given country.

        Parameters
        ----------
        data : pd.DataFrame
            The inflow data to be plotted.
        path : str
            The file path where the plot will be saved.
        country_code : str
            The code of the country for which the inflow data is being plotted.

        Returns
        -------
        None
        """
        data.plot(figsize=(12,5), label='Inflow Weekly')
        plt.title(f"Inflow in {country_code}")
        plt.xlabel('Time')
        plt.ylabel('Inflow (MWh)')
        plt.legend()
        
        plt.savefig(path, bbox_inches='tight')
        plt.show()
        self.logger.info(f'Save historical inflow fig for {country_code}--->Finished')

        return

        
         

    









