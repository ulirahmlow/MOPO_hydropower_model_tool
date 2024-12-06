# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:49:43 2024

@author: yil2
"""

import geopandas as gpd
from shapely.geometry import Polygon,LineString,Point
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from matplotlib.lines import Line2D
import os
import shutil
import logging
import warnings

class geo_process:
    
    def __init__(self):
        pass

    ########################connect the plants to the geo map
    def basin_process(
        self, 
        zone: gpd.GeoDataFrame, 
        Basins: gpd.GeoDataFrame, 
        JRC_zone: pd.DataFrame, 
        Rivers: gpd.GeoDataFrame, 
        country_code: str
    ) -> gpd.GeoDataFrame:
        """
        zone: geo dataframe of this zone area
        Basins: basins within this zone area
        JRC_geo: hydropower plants within this zone area

        process the basins so that below are included:
        1. Basins entirely within the area
        2. Basins in the border line whose downstream are within the area
    
        Returns
        -------
        basins_with: GeoDataFrame
            Basins with plants within the zone area
        """
    
        #################################################################################################
        ########seperate relative basins into inner basins and basins partially included
        logging.info(f"Processing basins started for region: {country_code}")
    
        try: 
            zone=zone.to_crs(Basins.crs)
            warnings.filterwarnings("ignore", message=".*Results from 'buffer' are likely incorrect.*")
            buffered_zone = zone.geometry.buffer(0.1)      # make sure the basins in the border line is not included as 'paritial' bains
            basins_border = Basins[~Basins.within(buffered_zone.unary_union)]  #basins in the border
            # within() a single geometry, return series of bool value
            basins_inner = Basins[Basins.within(buffered_zone.unary_union)]# basins inside the zone
            basins_border = basins_border.reset_index(drop=True)
            basins_inner = basins_inner.reset_index(drop=True)
            
            
            #################################################################################################
            #remove the basins without plants: we dont consider basins without plants
                
            ###########TODO: remove basins is tributary (pf code last digital is even ), means: keep those are not tributary.

            if 'index_right' in JRC_zone.columns: #JRC_zone used the inner join method, anything didnt intersect has been dropped already
                JRC_zone = JRC_zone.drop(columns=['index_right'])
            
        
            basins_with = basins_border[basins_border.geometry.apply(lambda polygon: JRC_zone.intersects(polygon).any())]
            basins_without = basins_border[~basins_border.index.isin(basins_with.index)]
            basins_border=basins_without #update the border basins: remove the basins with plants
        
            #################################################################################################
            #test if the next-down basins is within the zone-basins
        
            basins_downstream_within= basins_border[basins_border['NEXT_DOWN'].isin(basins_inner['HYBAS_ID'])]       # border basins whose downstream is in the zone area
        
            if not basins_downstream_within.empty:
        
                basins_exclude= basins_border[~basins_border.within(basins_downstream_within.unary_union)] 
                basins_include=Basins[~Basins.within(basins_exclude.unary_union)]
            else:

                basins_include=basins_inner
                
            basins_with=pd.concat([basins_include, basins_with], ignore_index=True)
            
        except Exception as e:
        
            logging.error(f"Error of processing basins {country_code}: {e}")
                
        return basins_with
    
    
    
    def zone_process(
        self, 
        zone_code: list[str], 
        pecd: gpd.GeoDataFrame, 
        lev07: gpd.GeoDataFrame, 
        eu_river: gpd.GeoDataFrame, 
        JRC: pd.DataFrame, 
        country_code: str
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        
        """
        Process the rivers, basins, and pecd to filter the zone level with zone code
    
        Parameters:
        zone_code (List[str]): the zone code for filtering
        pecd (gpd.GeoDataFrame): the pecd dataframe
        lev07 (gpd.GeoDataFrame): the lev07 dataframe
        eu_river (gpd.GeoDataFrame): the eu_river dataframe
        JRC (pd.DataFrame): the JRC dataframe
        country_code (str): the country code
        Returns:
        zone (gpd.GeoDataFrame): the filtered zone
        JRC_zone (gpd.GeoDataFrame): the filtered JRC zone
        Basins (gpd.GeoDataFrame): the filtered Basins
        Rivers (gpd.GeoDataFrame): the filtered Rivers
        """
        logging.info(f"Processing polygons started for region: {country_code}")
        
        zone = pecd[pecd['id'].isin(zone_code)]
        Basins=lev07[lev07.intersects(zone.unary_union)] #filter the lev07 basins according to the zone
        Rivers=eu_river[eu_river.geometry.intersects(zone.unary_union)]#intersects polygon and linestream
        Rivers=Rivers[Rivers['ORD_FLOW']<6]

        #########convert the csv into geo dataframe
        JRC['geometry']=JRC.apply(lambda row: Point(row['lon'], row['lat']),axis=1)
        JRC_geo=gpd.GeoDataFrame(JRC,geometry='geometry')
        JRC_geo=JRC_geo.set_crs('EPSG:4326')
        #zone=zone.to_crs(4326)
        JRC_zone=gpd.sjoin(JRC_geo, zone, how='inner',predicate='within')
        
    
       # logging.error(f"Error of processing polygons {country_code}: {e}")
        
    
        return zone, JRC_zone, Basins, Rivers
    
    def draw_geo(
        self, 
        zone: gpd.GeoDataFrame, 
        basins_include: gpd.GeoDataFrame, 
        JRC_zone: gpd.GeoDataFrame, 
        Rivers: gpd.GeoDataFrame, 
        zone_code: str
    ) -> None:
        """Draw the geo data with main rivers and basins within the zone

        Parameters:
        zone (gpd.GeoDataFrame): the zone
        basins_include (gpd.GeoDataFrame): the basins within the zone
        JRC_zone (gpd.GeoDataFrame): the JRC power plants within the zone
        Rivers (gpd.GeoDataFrame): the main rivers within the zone
        zone_code (str): the zone code

        Returns:
        None
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        zone.plot(ax=ax, facecolor='none', edgecolor='black')
        basins_include.plot(ax=ax, facecolor='#C9DCC4', edgecolor='#4f845c')
        Rivers.plot(ax=ax, facecolor='none', edgecolor='#599cb4')
        JRC_zone.plot(ax=ax, color='#c25759', markersize=10)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f"Main Rivers and Basins in Zone {zone_code}")
        legend_handles = [
        Line2D([0], [0], color='black', lw=2, label='Zone area'),
        Line2D([0], [0], color='#599cb4', lw=2, label='Main Rivers within Zone'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C9DCC4', markersize=10, label='Basins within Zone'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#c25759', markersize=10, label='Power Plants within Zone')
     ]

        # Add legend to the plot
        ax.legend(handles=legend_handles, loc='upper right')
        plt.show()

        return
    
    
    
    def save_geo(self, geo_file: gpd.GeoDataFrame, filepath: str) -> None:
        """
        Save the GeoDataFrame to the specified file path.

        Parameters:
        geo_file (gpd.GeoDataFrame): The GeoDataFrame to be saved.
        filepath (str): The path where the GeoDataFrame will be saved.

        Returns:
        None
        """
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)

        geo_file.to_file(filepath)
        print('Process and save geography data ---> Finished')

        return
    
    
    
    
    
    
    
    
    