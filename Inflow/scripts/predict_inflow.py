# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:57:38 2024

@author: yil2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os
import config

class predictor:
    
    def __init__(self, country_code, year):
        self.country_code = country_code
        self.year=year
        self.path=config.pred_data_path

        self.logger = logging.getLogger('regression model')
    
    def datetime_index(self, data, time_column):
        """
        tranfer the index of target data into datetime
        time_column: the column name to be the index
        
        """
        
        data.set_index(pd.to_datetime(data[time_column]), inplace=True)
        data=data.drop([time_column], axis=1)
        data.index.name='time'
        data.index=data.index.tz_localize(None)
        
        
        return data
    
    
    def data_preprocess(
        self, runoff: pd.DataFrame, inflow: pd.DataFrame, frequency: str = "W-SUN", aggregation: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses runoff and inflow data by resampling and aligning time indices.

        Parameters:
            runoff (pd.DataFrame): The runoff data to be preprocessed.
            inflow (pd.DataFrame): The inflow data to be preprocessed.
            frequency (str, optional): The frequency for resampling the data. Defaults to 'W-SUN'.
            aggregation (bool, optional): If True, aggregates the runoff data by summing across columns.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - X (pd.DataFrame): The preprocessed and optionally aggregated runoff data.
                - y (pd.DataFrame): The resampled inflow data aligned with the runoff data.
        """
        runoff = runoff.resample(frequency).sum()
        inflow = inflow.resample(frequency).sum()

        start_time = max(runoff.index[0], inflow.index[0])
        end_time = min(runoff.index[-1], inflow.index[-1])
        self.logger.info(f"The time range of the training data is: {start_time} to {end_time}")

        inflow = inflow.loc[start_time:end_time]
        runoff = runoff.loc[start_time:end_time]

        if aggregation:
            X = runoff.sum(axis=1)
        else:
            X = runoff

        y = inflow

        return pd.DataFrame(X), pd.DataFrame(y)
        
        
        
    
    
    
    def regression(self, X: pd.DataFrame, y: pd.DataFrame, image_path: str, positive: bool = True)->tuple:
    
        """
        Split the data into training and testing set, perform linear regression 
        and predict the inflow for the given year.

        Parameters
        ----------
        X: pd.DataFrame
            The predictor data.
        y: pd.DataFrame
            The response data.
        positive: bool, optional
            Whether to use positive or regular LinearRegression. Default to True.

        Returns
        -------
        tuple
            A tuple containing the predicted inflow, historical inflow, 
            the regression model.
        """

        test_year=self.year
        X_test=X[(X.index.year>=test_year)&(X.index.year<test_year+1)]
        y_test=y[(y.index.year>=test_year)&(y.index.year<test_year+1)]
        X_train=X.drop(index=X_test.index, axis=0)
        y_train=y.drop(index=y_test.index, axis=0)
        
        self.logger.info(f'Training shapes: {len(X_train)}')
        self.logger.info(f'Testingg shapes: {len(X_test)}')

        regr=LinearRegression(fit_intercept=False,positive=positive)
        regr.fit(X_train,y_train)
        
       
        y_pred = regr.predict(X_test)
        y_pred=pd.DataFrame(y_pred)
        y_pred.set_index(y_test.index, inplace=True)

        y_model=regr.predict(X)
        y_model=pd.DataFrame(y_model)
        y_model.set_index(X.index, inplace=True)
        
        
        plt.figure(figsize=(12,5))
        plt.plot(y_model, color='#c25759', label='predicted inflow')
        plt.plot(y, color='#599cb4', linestyle='dashed', label='historical inflow')
        plt.xlabel('Time')
        plt.ylabel('Inflow /MWh')
        plt.title(f'{self.country_code} predicted inflow')
        plt.savefig(f"{image_path}//{self.country_code}_regression.pdf", format="pdf", bbox_inches="tight")
        plt.legend()
     
        plt.show()
    

        #regr.coef_=pd.DataFrame(regr.coef_)

    
        self.logger.info(f'Predicting {self.year} inflow ---> Finished')
        
        return pd.DataFrame(y_pred), pd.DataFrame(y_test), regr
    
    def evaluation(self, y_pred: pd.DataFrame, y_test: pd.DataFrame)->None:
        
        """
        Evaluate the predicted inflow by calculating the mean squared error, mean absolute error, 
        root mean squared percentage error and R-squared.

        Parameters
        ----------
        y_pred : pd.DataFrame
            The predicted inflow data.
        y_test : pd.DataFrame
            The historical inflow data.
        """
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)/(y_test.mean())
    
        self.logger.info(f'The MSE of predicted value is:{mse}')
        self.logger.info(f'The MAE of predicted value is:{mae}')
        self.logger.info(f'The RMSE of predicted value is:{rmse}')
        self.logger.info(f'The R2 of predicted value is:{r2}')
        
        return
    
    def desample(self, input_data: pd.DataFrame, resolution: str)->pd.DataFrame:

        """
        Desamples the input data to a given resolution.

        Parameters
        ----------
        input_data : pd.DataFrame
            The input data to be desampled.
        resolution : str
            The target resolution. Can be either 'daily' or 'hourly'.

        Returns
        -------
        tuple
            A DataFrame containing the desampled data.

        """
        
        if resolution=='daily':
            input_data=input_data.resample('D').ffill().div(7)

            
        elif resolution=='hourly':
            input_data=input_data.resample('h').ffill().div(168)
          
        return pd.DataFrame(input_data)
    
    
    def save_fig(self, y_pred: pd.DataFrame, y_test: pd.DataFrame, period: int, image_path: str)->None:
        
        if period==30:
            freq='monthly'
        elif period==7:
            freq='weekly'
        elif period==24:
            freq='daily'
        
        plt.figure(figsize=(12,5))
        plt.plot(y_pred, color='#c25759', label='predicted inflow')
        plt.plot(y_test, color='#599cb4', linestyle='dashed', label='historical inflow')
        plt.xlabel('Time')
        plt.ylabel('Inflow /MWh')
        plt.title(f'{self.country_code}_hourly predicted inflow')
        plt.legend()
      
        plt.savefig(f"{image_path}//{self.country_code}_{self.year}_inflow_hourly.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        
        self.logger.info(f"Saving {self.year} predicted inflow figure for region: {self.country_code}--->Finished")
        
        return
    def save_fig_longterm(self, y_pred: pd.DataFrame, image_path: str) -> None:
        """
        Saves a plot of the long-term predicted inflow.

        Parameters
        ----------
        y_pred : pd.DataFrame
            The predicted inflow data.
        image_path : str
            The file path where the plot will be saved.

        Returns
        -------
        None
        """
        plt.figure(figsize=(12,5))
        plt.plot(y_pred, color='#c25759', label='long-term predicted inflow')
        plt.xlabel('Time')
        plt.ylabel('Inflow /MWh')
        plt.title(f'{self.country_code}_hourly predicted inflow')
        plt.legend()
      
        plt.savefig(f"{image_path}//{self.country_code}_1991_2021_inflow_hourly.pdf", format="pdf", bbox_inches="tight")
        #plt.show()
        
        self.logger.info(f"Saving long-term predicted inflow figure for region: {self.country_code}--->Finished")
        
        return
        
    def save_file(self, y_pred: pd.DataFrame, file_name: str) -> None:
        """
        Saves the predicted inflow into a CSV file.

        Parameters
        ----------
        y_pred : pd.DataFrame
            The predicted inflow data.
        file_name : str
            The file name of the CSV file to be saved.

        Returns
        -------
        None
        """
        #file_name=f'/{self.country_code}_{self.year}_LR_inflow.csv'
        file_path=os.path.join(self.path, file_name)
        y_pred.to_csv(file_path,sep=';')
        
        self.logger.info(f"Saving predicted inflow into csv for region: {self.country_code}--->Finished")
        
         
        return
    
    def create_lags(self, input_X: pd.DataFrame, lags: int)->pd.DataFrame:
        """
        Create lagged features for the given input data.

        Parameters
        ----------
        input_X : pd.DataFrame
            The input data to be lagged.
        lags : int
            The number of lags to be created.

        Returns
        -------
        pd.DataFrame
            The DataFrame with lagged features.
        """
        input_X_lags=pd.DataFrame()
        for i in range(1,lags+1):
            X_lags=input_X.shift(periods=i)
            input_X_lags=pd.concat([input_X_lags, X_lags], axis=1)

        output_X=pd.concat([input_X, input_X_lags], axis=1)
        output_X=output_X.dropna()
        
        return pd.DataFrame(output_X)