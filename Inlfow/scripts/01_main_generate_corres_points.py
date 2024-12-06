# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:32:54 2024

@author: yil2
"""
import geopandas as gpd
import pandas as pd
import os
import logging
from geo_process import*
import config

country_code=config.country_code
grids_path=config.era5_grids_path
era5_points=gpd.read_file(grids_path)

def aggregate_corres():
    basins_filepath=config.filtered_basin_filepath
    basins_filename=f'\\{country_code}_basins_lev07.shp'
    corres_points_path=config.corres_points_path

    country_map_filepath=config.country_map_path
    country_code_excel=pd.read_excel(country_map_filepath, index_col='Input_code', 
                                    usecols=['Input_code','Entsoe','eSett','PECD2'])
    country_code_excel.index = country_code_excel.index.str.strip() ####remove spaces
    country_code_list=country_code_excel.index
    entsoe_code, eSett_code, PECD2_level=(country_code_excel.loc[country_code])
    zone_code=PECD2_level.split(',')


    process=geo_process()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    if not os.path.isdir(basins_filepath):
        os.makedirs(basins_filepath)
        logger.info(f"Processing geography for region: {country_code}---> Start")
        
        lev07_basins=gpd.read_file(config.basin_filepath)
        ########Without considering rivers at this stage#################################
        eu_river=gpd.read_file(config.river_filepath)
        onshore= gpd.read_file(config.onshore_filepath)
        pecd=onshore[onshore['level']=='PECD2']
        JRC_plants=pd.read_csv(config.jrc_filepath)
        zone_area, plants_inzone, basins_inzone, rivers_inzone=process.zone_process(
            zone_code, pecd, lev07_basins, eu_river, JRC_plants,country_code)
        
        basins_filtered=process.basin_process(
            zone_area, basins_inzone, plants_inzone, rivers_inzone,country_code)
        
        process.draw_geo(zone_area, basins_filtered, plants_inzone,rivers_inzone, country_code)
        
        process.save_geo(basins_filtered, basins_filepath)
        logger.info(f"Processing geography for region: {country_code}--->Finished")

    else:
        
        basins_filtered=gpd.read_file(basins_filepath + basins_filename)

        
    logger.info(f"Reading geography for region: {country_code}--->Finished")    
        
    corres_in_basins = gpd.sjoin(era5_points,basins_filtered, how='right',predicate='within' )
    ########export data to Cores ready files
    unique_basins = corres_in_basins['HYBAS_ID'].unique()
    basin_mapping = {basin: f'{country_code}_HYBAS{str(i+1).zfill(2)}' for i, basin in enumerate(unique_basins)}
    corres_in_basins['HYBAS_ID'] = corres_in_basins['HYBAS_ID'].map(basin_mapping)
    corres_in_basins=corres_in_basins.dropna()
    corres_points=pd.DataFrame({
                'Latitude': corres_in_basins['latitude'],
                'Longitude':corres_in_basins['longitude'],
                'Region':corres_in_basins['HYBAS_ID'],
                'Weight_group': 1.0,
                'Area':country_code
    })
    
    corres_points.to_excel(corres_points_path, index=False)
    logger.info(f"Generate CorRES points for region: {country_code}--->Finished")


if __name__ == '__main__':
    aggregate_corres()