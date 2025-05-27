import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import os
from glob import glob


class GeoProcess:    
    def __init__(self, config_obj, path_obj):
        self.config_obj = config_obj
        self.path_obj = path_obj

    def sjoin_gdf(self, gdf1, gdf2):
        gdf=gdf1.sjoin(gdf2)
        
        if 'index_right' in gdf.columns:
            gdf=gdf.drop(columns='index_right')

        if 'index_left' in gdf.columns:
            gdf=gdf.drop(columns='index_left')

        return gdf
    
    def read_onshore_zone(self, file_path, zone_code):
        onshore =gpd.read_file(file_path)
        pecd=onshore[onshore['level']=='PECD2']
        zone = pecd[pecd['id'].isin(zone_code)]
        return zone


class GeoBasins(GeoProcess):
    def __init__(self, config_obj, path_obj):
        super().__init__(config_obj, path_obj)

    # def sjoin_gdf(self, gdf1, gdf2):
    #     gdf=gdf1.sjoin(gdf2)
        
    #     if 'index_right' in gdf.columns:
    #         gdf=gdf.drop(columns='index_right')

    #     if 'index_left' in gdf.columns:
    #         gdf=gdf.drop(columns='index_left')

    #     return gdf

    def find_res_basin(self, pfcode, res_location, bas_lev7, bas_lev6, bas_lev5):
        #pfcode=[]
        for location in res_location['geometry']:
            found=False
            for _, bas in bas_lev7.iterrows(): 
                polygon = bas['geometry']
                id_code = bas['PFAF_ID']
                if location.within(polygon):
                    if id_code %2 ==0:
                        pfcode.append(id_code)
                        #print('found even basin level7')
                        found=True
                        break
                    else:
                        id_code_str=str(id_code)[:-1]
                        for all_code in bas_lev7['PFAF_ID']:
                            if str(all_code)[3:-2]==id_code_str:
                                print(id_code_str)
                                print(location)
                                if all_code % 20 ==0:
                                    pfcode.append(all_code)
                                    found=True
                                    #print('found same code basin level7')
                                    break
                                else: 
                                    found=False
            if not found:    
                #print('search basin level6')                
                for _, bas_6 in bas_lev6.iterrows():
                    id_6= bas_6['PFAF_ID']
                    polygon = bas_6['geometry']
                    if location.within(polygon):
                        #print('find basin 6 location')
                        if id_6 % 2 == 0:
                            pfcode.append(id_6)
                            found=True
                            #print('found basin level6')
                            break
                        else:
                            id_6_str=str(id_6)[:-1]
                            for all_code_6 in bas_lev6['PFAF_ID']:
                                if str(all_code_6)[3:-2]==id_6_str:
                                    if all_code_6 %20 ==0:
                                        pfcode.append(all_code_6)
                                        found=True
                                        #print('found basin level6')
                                        break
                                    else:
                                        #print('error:didnt have basin 7')
                                        found=False
            if not found:
                for _, bas_5 in bas_lev5.iterrows():
                    id_5= bas_5['PFAF_ID']
                    polygon = bas_5['geometry']
                    if location.within(polygon):
                        if id_5 % 2 == 0:
                            pfcode.append(id_5)
                            found=True
                            #print('found basin level5')
                            break
                        else:
                            id_5_str=str(id_5)[:-1]
                            for all_code_5 in bas_lev5['PFAF_ID']:
                                if str(all_code_5)[3:-2]==id_5_str:
                                    if all_code_5 %20 ==0:
                                        pfcode.append(all_code_5)
                                        found=True
                                        #print('found basin level5')
                                        break

        return pfcode


    def find_next_downstream(self, bas, bas_gdf):
        up_bas=bas
        all_bas=up_bas
        for i in range(len(bas_gdf)):
            up_bas=bas_gdf[bas_gdf['NEXT_DOWN'].isin(up_bas['HYBAS_ID'])]
            all_bas=pd.concat([all_bas, up_bas])
            if len(up_bas)==0:
                break
        
        unique_id=pd.concat([all_bas['HYBAS_ID'], bas['HYBAS_ID']]).drop_duplicates(keep=False, ignore_index=True)
        upper_bas=bas_gdf[bas_gdf['HYBAS_ID'].isin(unique_id)]
        return upper_bas

    def cds_grid_output(self, era5_grids, polygon, zone_code, file_path):

        grid_in_polygon = gpd.sjoin(era5_grids,polygon, how='right',predicate='within' )
        ########export data to Cores ready files
        unique_basins = grid_in_polygon['HYBAS_ID'].unique()
        basin_mapping = {basin: f'{zone_code}_HYBAS{str(i+1).zfill(2)}' for i, basin in enumerate(unique_basins)}
        grid_in_polygon['HYBAS_ID'] = grid_in_polygon['HYBAS_ID'].map(basin_mapping)
        grid_in_polygon=grid_in_polygon.dropna()
        cds_points=pd.DataFrame({
                    'Latitude': grid_in_polygon['latitude'],
                    'Longitude':grid_in_polygon['longitude'],
                    'Region':grid_in_polygon['HYBAS_ID'],
                    'Weight_group': 1.0,
                    'Area':zone_code
        })
        cds_points.to_excel(file_path, index=False)

    def save_cds_grids(self):
        code = self.config_obj.country_code
        map = self.config_obj.map
        config = self.config_obj.config
        path_dict = self.path_obj.path_dict

        if not path_dict['cds_points_path'].exists():
            print('Processing upstream basins polygon data.')
            self.save_cds_grids_main(code, map, config, path_dict)
        else:
            print('Upstream basins polygon data already exists. Skipping geo-processing.')


    def save_cds_grids_main(self, code, map, config, path_dict):

        #___________________________________________load reservoir and lake data________________________________

        res=pd.read_csv(config['res_path'], usecols=['LisfloodX', 'LisfloodY'])
        lake=pd.read_csv(config['lake_path'], usecols=['LisfloodX', 'LisfloodY'])
        zone = self.read_onshore_zone(config['onshore_path'], map[code]['PECD2'])

        res_gpd=gpd.GeoDataFrame(res, geometry=gpd.points_from_xy(res['LisfloodX'], res['LisfloodY']), crs=zone.crs)
        lake_gpd=gpd.GeoDataFrame(lake, geometry=gpd.points_from_xy(lake['LisfloodX'], lake['LisfloodY']), crs=zone.crs)

        zone_res=self.sjoin_gdf(res_gpd,zone)
        zone_lake=self.sjoin_gdf(lake_gpd,zone)

        bas_lev7=gpd.read_file(config['bas_lev7_path'])
        bas_lev6=gpd.read_file(config['bas_lev6_path'])
        bas_lev5=gpd.read_file(config['bas_lev5_path'])


        #_____________________________________________determine reservoir and lake pfcode_______________________
        pfcode=[]
        pfcode=self.find_res_basin(pfcode, zone_res, bas_lev7, bas_lev6, bas_lev5)
        pfcode=self.find_res_basin(pfcode, zone_lake, bas_lev7, bas_lev6, bas_lev5)
        upstream_basins_lev7=bas_lev7[bas_lev7['PFAF_ID'].isin(pfcode)]
        upstream_basins_lev6=bas_lev6[bas_lev6['PFAF_ID'].isin(pfcode)]
        upstream_basins_lev5=bas_lev5[bas_lev5['PFAF_ID'].isin(pfcode)]

        # TODO: Check here
        if config['upper_basin'] is True: 
            upper_lev7=self.find_next_downstream(upstream_basins_lev7, bas_lev7)
            upper_lev6=self.find_next_downstream(upstream_basins_lev6, bas_lev6)
            upper_lev5=self.find_next_downstream(upstream_basins_lev5, bas_lev5)
            if upper_lev7.is_empty.all() and upper_lev6.is_empty.all() and upper_lev5.is_empty.all():
                print('No upper basins found')
            #TODO: if no upper basins then update the geo_range to local

            #_______________________________________________draw figure_______________________________________________
            figure, axes = plt.subplots(figsize=(10, 6), nrows=1, ncols=3,sharey=True)
            levels = [(upstream_basins_lev7, 'mintcream', 'lightskyblue', 'Level 7 Sub-basins'),
                    (upstream_basins_lev6, 'honeydew', 'seagreen', 'Level 6 Sub-basins'),
                    (upstream_basins_lev5, 'lavender', 'mediumpurple', 'Level 5 Sub-basins')]
            upper_levels=[(upper_lev7, 'lavender', 'mediumpurple'),
                        (upper_lev6, 'mintcream', 'lightskyblue'),
                        (upper_lev5, 'honeydew', 'seagreen')]

            for ax, (basins, facecolor, edgecolor, title), (up_bas, fc, ec) in zip(axes, levels, upper_levels):
                ax.set_aspect('equal', adjustable='box')
                zone.plot(ax=ax, facecolor='none', edgecolor='black')

                if not basins.is_empty.all():
                    basins.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.5)
                if not up_bas.is_empty.all():
                    up_bas.plot(ax=ax, facecolor=fc, edgecolor=ec, linewidth=0.5)

                if not zone_res.is_empty.all():
                    zone_res.plot(ax=ax, facecolor='none', edgecolor='orangered', markersize=1, label='Reservoirs')
                if not zone_lake.is_empty.all():
                    zone_lake.plot(ax=ax, facecolor='none', edgecolor='lime', markersize=1, label='Lakes')
                ax.legend(loc='lower right')
                ax.set_title(title, fontsize=14)
                ax.set_xlabel('Longitude', fontsize=12)

            axes[0].set_ylabel('Latitude', fontsize=12)
            plt.savefig(path_dict['up_basins_png_path'], dpi=300)
            plt.show()
            up_basins=pd.concat([upstream_basins_lev7, upstream_basins_lev6, upstream_basins_lev5,
                                upper_lev7, upper_lev6, upper_lev5])

            

        else:

            figure, axes = plt.subplots(figsize=(10, 6), nrows=1, ncols=3,sharey=True)
            levels = [(upstream_basins_lev7, 'mintcream', 'lightskyblue', 'Level 7 Sub-basins'),
                    (upstream_basins_lev6, 'honeydew', 'seagreen', 'Level 6 Sub-basins'),
                    (upstream_basins_lev5, 'lavender', 'mediumpurple', 'Level 5 Sub-basins')]

            for ax, (basins, facecolor, edgecolor, title) in zip(axes, levels):
                ax.set_aspect('equal', adjustable='box')
                zone.plot(ax=ax, facecolor='none', edgecolor='black')
                if not basins.is_empty.all():
                    basins.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.5)
                if not zone_res.is_empty.all():
                    zone_res.plot(ax=ax, facecolor='none', edgecolor='orangered', markersize=1, label='Reservoirs')
                if not zone_lake.is_empty.all():
                    zone_lake.plot(ax=ax, facecolor='none', edgecolor='lime', markersize=1, label='Lakes')
                ax.legend(loc='lower right')
                ax.set_title(title, fontsize=14)
                ax.set_xlabel('Longitude', fontsize=12)

            axes[0].set_ylabel('Latitude', fontsize=12)
            plt.savefig(path_dict['up_basins_png_path'], dpi=300)
            plt.show()
            up_basins=pd.concat([upstream_basins_lev7, upstream_basins_lev6, upstream_basins_lev5])

        #_______________________________________________get era5 points____________________________________________
        
    
        era5_points=gpd.read_file(path_dict['geo_grid_path'])

        self.cds_grid_output(era5_points, up_basins, code, path_dict['cds_points_path'])

        print(f"Generate ERA5 grids box for region: {code}--->Finished")


class GeoStreams(GeoProcess):
    def __init__(self, config_obj, path_obj):
        super(GeoStreams, self).__init__(config_obj, path_obj)


    def save_river_id_main(self, code, map, config, path_dict):

    
        onshore = gpd.read_file(config['onshore_path'])
        pecd=onshore[onshore['level']=='PECD2']
        #zone_code = code
        pecd_code = map[code]['PECD2']
        zone = pecd[pecd['id'].isin(pecd_code)]
        
        
        res=pd.read_csv(config['res_path'], usecols=['LisfloodX', 'LisfloodY'])
        lake=pd.read_csv(config['lake_path'], usecols=['LisfloodX', 'LisfloodY'])

        res_gpd=gpd.GeoDataFrame(res, geometry=gpd.points_from_xy(res['LisfloodX'], res['LisfloodY']), crs=zone.crs)
        lake_gpd=gpd.GeoDataFrame(lake, geometry=gpd.points_from_xy(lake['LisfloodX'], lake['LisfloodY']), crs=zone.crs)
        zone_res=self.sjoin_gdf(res_gpd,zone)
        zone_lake=self.sjoin_gdf(lake_gpd,zone)

        
        eu_river_files = str(config['eu_stream_dir'])+ "\\*.gpkg"
        paths=sorted(glob(eu_river_files))
        river_gdf = pd.concat([gpd.read_file(file_path) for file_path in paths], ignore_index=True)
        river_proj=river_gdf.to_crs(zone.crs)

        res_buf=zone_res.copy()
        lake_buf=zone_lake.copy()
        res_buf['geometry']=res_buf['geometry'].buffer(0.03)
        lake_buf['geometry']=lake_buf['geometry'].buffer(0.03)
        #TODO: check Userwarning
        river_with_res=self.sjoin_gdf(river_proj, res_buf).drop_duplicates(['LINKNO'], keep='first')
        river_with_lake=self.sjoin_gdf(river_proj, lake_buf).drop_duplicates(['LINKNO'], keep='first')
        #river_zone=self.sjoin_gdf(river_proj, zone)

        fig, ax = plt.subplots(figsize=(10, 10))
        zone.plot(ax=ax, facecolor='none', edgecolor='black')
        if not river_with_res.is_empty.all():
            river_with_res.plot(ax=ax, facecolor='none', edgecolor='#599cb4')
        if not river_with_lake.is_empty.all():
            river_with_lake.plot(ax=ax, facecolor='none', edgecolor='#f1c40f')
        if not zone_res.is_empty.all():
            zone_res.plot(ax=ax, color='r')
        if not zone_lake.is_empty.all():
            zone_lake.plot(ax=ax, c='green')
        plt.savefig(path_dict['rivers_png_path'])

        river_with_res.LINKNO.to_csv(path_dict['res_river_id_path'], index=False)
        river_with_lake.LINKNO.to_csv(path_dict['lake_river_id_path'], index=False)

        
    def save_river_id(self):
        code = self.config_obj.country_code
        map = self.config_obj.map
        config = self.config_obj.config
        path_dict = self.path_obj.path_dict

        if os.path.exists(path_dict['res_river_id_path']) or os.path.exists(path_dict['lake_river_id_path']):
            print('River ID data already exists locally.  Skipping geo-processing.')
        else:
            print(f'Processing {code} river data...')
            self.save_river_id_main(code, map, config, path_dict)
