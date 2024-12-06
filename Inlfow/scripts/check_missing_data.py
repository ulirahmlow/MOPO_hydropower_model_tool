# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:14:29 2024

@author: yil2
"""

import pandas as pd
import logging

class Check_fill_data:
    def __init__(self):
        self.logger = logging.getLogger('check data')
        #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
    def create_date_range(self,start_date:str, end_date:str, freq='h') ->pd.DataFrame:
        """

        Parameters
        ----------
        start_date : string
            start time of a daterange
        end_date : string
            end time of a daterange
        freq: frequency of this daterange

        Returns
        -------
        a date range with sepecifc frequency

        """
        return pd.date_range(start_date, end_date, freq=freq)
    
    def check_missing_data(self, input_data:pd.DataFrame, date_range:pd.DatetimeIndex, freq='h')->pd.DataFrame:
        """
        

        Parameters
        ----------
        input_data : pd.DataFrame
            The input_data for data checking
        date_range : pd.DatetimeIndex
            The referenced date range

        Returns
        -------
        The expcted data with correct date range

        """
       
        
       
        if input_data.index[-1]!=date_range[-1]:
            self.logger.info('Incorrect end time')
            input_data.loc[date_range[-1]]=input_data.iloc[-1]
            missed_length=len(date_range)-len(input_data.index)+1
            self.logger.info(f'Number of missed data:{missed_length}')

            
        if input_data.index[0]!=date_range[0]:
            self.logger.info('Incorrect start time')
            input_data.loc[date_range[0]]=input_data.iloc[0]
            missed_length=len(date_range)-len(input_data.index)+1
            self.logger.info(f'Number of missed data:{missed_length}')


                
        if len(input_data.index)!=len(date_range):
            missed_length=len(date_range)-len(input_data.index)
            self.logger.info(f'Number of interpolated data:{missed_length}', )
            input_data=input_data.resample(freq).asfreq()
            input_data=input_data.interpolate(method='linear')
            self.logger.info(f'Data has been fixed: new length= {len(input_data.index)}')
            
        else:
            self.logger.info('The index of data is already continous')
            
        input_data=pd.DataFrame(input_data)
        
        return input_data
        
    def check_duplicate_data(self, input_data:pd.DataFrame)->pd.DataFrame:
        
        """
        Remove duplicate entries from the input DataFrame based on the index.

        Parameters
        ----------
        input_data : pd.DataFrame
            The DataFrame to check for duplicates.

        Returns
        -------
        pd.DataFrame
            A DataFrame with duplicate index entries removed, keeping the first occurrence.
        """
        output_data=input_data[~input_data.index.duplicated(keep='first')]
        self.logger.info(f'Duplicated data length: {len(input_data.index)-len(output_data.index)}')

        return pd.DataFrame(output_data)
    
    def check_negative_data(self, data:pd.DataFrame)->pd.DataFrame:
        """
        Fill the zero values in the data by the mean of the previous and next values.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The data with zero values replaced by the mean of the previous and next values.
        """
        if (data<0).any().any():
            self.logger.info('There are negative values in the data')
            data=data.mask(data<=0)
            pd.set_option('future.no_silent_downcasting', True)
            data_ffilled = data.ffill().infer_objects(copy=False)
            data_bfilled = data.bfill().infer_objects(copy=False)
            mean_filled_value=(data_bfilled+data_ffilled)/2
            data.fillna(mean_filled_value,inplace=True)
        else:
            self.logger.info('There are no negative values in the data')

        return pd.DataFrame(data)
       
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    