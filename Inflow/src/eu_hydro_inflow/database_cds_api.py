import cdsapi
import pandas as pd
import xarray as xr
import os
from glob import glob
import geoglows


class DatabaseCds:

    #DATASET = "reanalysis-era5-land"
    REQUEST_YEAR=["2015","2016","2017","2018","2019","2020","2021", "2022", "2023","2024"]
    REQUEST_MONTH=["01","02","03","04","05","06","07","08","09","10","11","12"]
    
    def __init__(self, config_obj, path_obj):
        self.config = config_obj.config
        self.path_dict = path_obj.path_dict
        self.country_code = config_obj.country_code
        self.dataset=self.config['cds_source']
        self.geo_range = path_obj.geo_range
        

    def cds_read(self, path):
        for year in DatabaseCds.REQUEST_YEAR:
            for month in DatabaseCds.REQUEST_MONTH:
                #FIXME: if single_level: "product_type": ["reanalysis"],
                request = {
                    "variable": [
                        "snowmelt",
                        "sub_surface_runoff",
                        "surface_runoff",
                        "total_precipitation"
                        ""
                    ],
                    "year": year,
                    "month": month,
                    "day": [
                        "01", "02", "03","04", "05", "06", "07", "08", "09", "10", "11", "12",
                        "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
                        "25","26","27","28","29","30","31"
                    ],
                    "time": [
                        "00:00"
                    ],
                    "data_format": "netcdf4",
                    "download_format": "unarchived",
                    "area": [72, -25, 34, 45]
                }

                client = cdsapi.Client()
                client.retrieve(self.dataset, request, os.path.join(path,f"{year}_{month}_00utc.nc"))
                print(f"Retrieve {year}_{month} successfully")  #TODO: REMOVE os.path.join???


    def read_netcdfs(self, files, dim):
        
        def process_one_path(path):
            ds = xr.open_dataset(path, engine="netcdf4")
            ds_copy = ds.load()  # Force load all data into memory
            ds.close()           # Manually close the file
            return ds_copy
        
        paths=sorted(glob(files))
        datasets=[process_one_path(path) for path in paths]
        combined=xr.concat(datasets,dim)
        return combined


    def extract_values(self, row, variable_name, ds):
        lat, lon = row['Latitude'], row['Longitude']
        val = ds[variable_name].sel(latitude=lat, longitude=lon, method='nearest').values
        return val


    def aggregate_value(self, variable_name, data, time_range):
        if variable_name == 't2m':
            output=data.groupby(['Region', 'Weight_group']).agg({'t2m':'mean'}).reset_index()
        else:
            output=data.groupby(['Region', 'Weight_group']).agg({variable_name:'sum'}).reset_index()

        reshape_dict={}
        for idx, row in output.iterrows():
            reshape_dict[row['Region']]=row[variable_name]
        
        df_reshaped=pd.DataFrame(reshape_dict, index=pd.to_datetime(time_range))

        return df_reshaped

    def __save_era5_main(self, cds_path, country_code, cds_points_path, predictors, history_data_path, geo_range):
        dataset = self.dataset

        if len(os.listdir(cds_path))==0:
            print(f"Downloading data from {dataset} to {cds_path}")
            print(f"Note: it needs hours")
            self.cds_read(cds_path)
        else: 
            print(f"{dataset} is already retrieved to {cds_path}")

        print(f"Read {dataset} data for {country_code}")

        predictor_paths = {
            var: history_data_path / f"{country_code}_{geo_range}{dataset}_{var}.csv" 
            for var in predictors
        }

        missing_predictors = [var for var, path in predictor_paths.items() if not path.exists()]

        if missing_predictors:
            data=self.read_netcdfs(str(cds_path) + "\\*.nc" , "valid_time")
            grids=pd.read_excel(cds_points_path)

            for var in missing_predictors:
                print(f"{country_code} {dataset} {var} is missing. Generating data...")
                grids[var]= grids.apply(lambda row: self.extract_values(row, var, data), axis=1)
                data_agg = self.aggregate_value(var, grids,pd.date_range(data['valid_time'].values[0], data['valid_time'].values[-1], freq='d').shift(-1))
                data_agg.to_csv(predictor_paths[var])
                print(f"{country_code} {dataset} {var} is saved to csv")

        else:
            print(f"{country_code} {dataset} data already exists locally. Skipping data fetching")

    def save_era5(self):

        cds_path = self.path_dict['cds_dataset_path']
        country_code = self.country_code
        cds_points_path = self.path_dict['cds_points_path']
        history_data_path = self.path_dict['history_data_path']
        predictors = self.config['predictors']
        self.__save_era5_main(cds_path, country_code, cds_points_path, predictors, history_data_path, self.geo_range)



class Databasegeoglows:
    def __init__(self, config_obj, path_obj):
        self.config = config_obj.config
        self.path_dict = path_obj.path_dict
        self.country_code = config_obj.country_code

    def save_streamflow(self):
            if os.path.exists(self.path_dict['res_streamflow_path']) or os.path.exists(self.path_dict['lake_streamflow_path']):
                print('Discharge data already exists locally. Skipping data fetching')

            else:
                print(f'Fetching streamflow data for {self.country_code}')
                res_river_id =pd.read_csv(self.path_dict['res_river_id_path'])
                lake_river_id=pd.read_csv(self.path_dict['lake_river_id_path'])
                
                res_streamflow = geoglows.data.retrospective(res_river_id['LINKNO'].tolist())
                lake_streamflow = geoglows.data.retrospective(lake_river_id['LINKNO'].tolist())
                res_streamflow.to_csv(self.path_dict['res_streamflow_path'], sep=';')
                lake_streamflow.to_csv(self.path_dict['lake_streamflow_path'], sep=';')
                print(f"{self.country_code} streamflow is saved to csv") 
