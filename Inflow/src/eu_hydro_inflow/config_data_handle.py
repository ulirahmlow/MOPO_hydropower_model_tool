from pathlib import Path
import toml
import sys

class ConfigData:

    def __init__(self):
        #FIX:add read only property
        self.__PATH_USER_CONFIG = Path(__file__).parent.parent / 'config_data' / 'user_config.toml'
        self.__PATH_COUNTRY_MAP = Path(__file__).parent.parent / 'config_data' / 'country_map.toml'
        self.config = {}
        self.map = {}
        self.country_code = ''
        self.hydro_type = ''
        self.__load_config()

    def __load_config(self):
        with open(self.__PATH_USER_CONFIG, 'r') as file:
            self.config = toml.load(file)
        with open(self.__PATH_COUNTRY_MAP, 'r') as file:
            self.map = toml.load(file)

    def args_check(self, args):
        # breakpoint()
        if args.input:
            country_code = args.input
        else:
            country_code = self.config['country_code']
        print(f'Selected Counrty is [{country_code}]')

        # Check country code
        if country_code not in self.map:
            print(f'Input country code: {country_code} is not available!')
            sys.exit(1)

        if args.type:
            hydro_type = args.type
        else:
            hydro_type = self.config['hydro_type']

        #Check for hydro_type and country_code matching
        if hydro_type == 'hdam' and self.map[country_code]['hdam_type_support']:
            print(f'Input country code: {country_code} Select hydro type : {hydro_type}')

        elif hydro_type == 'hror' and self.map[country_code]['hror_type_support']:
            print(f'Input country code: {country_code} Select hydro type : {hydro_type}')

        else:
            print(f'Input country code: [{country_code}] Select hydro type : [{hydro_type}] are not supported!')
            sys.exit(1)

        self.country_code = country_code
        self.hydro_type = hydro_type
        


class FetchPath:

    def __init__(self, config_obj):
        self.path_dict = {}
        self.__spine_gen_dir = []
        self.geo_range = None
        self.__set_file_path(config_obj)
        self.__create_dir()
        

    def __set_file_path(self, config_obj):
        country_code = config_obj.country_code
        geo_dir = Path(config_obj.config['geo_dir'])
        self.path_dict['basin_filepath'] = geo_dir / 'Hydrobasins' / 'hybas_eu_lev07_v1c.shp'
        self.path_dict['river_filepath'] = geo_dir / 'Hydrorivers' / 'HydroRIVERS_v10_eu_shp' / 'HydroRIVERS_v10_eu.shp'
        self.path_dict['onshore_filepath'] = geo_dir / 'onshore.geojson'
        self.path_dict['jrc_filepath'] = geo_dir / 'jrc-hydro-power-plant-database.csv'
        self.path_dict['filtered_basin_filepath'] = geo_dir / 'basins_filtered' / (country_code + '_basins_lev07')
        
        if config_obj.config['upper_basin']: 
            self.geo_range='upper_'
        else:
            self.geo_range=''
        
        # weather_dir = Path(config_obj.config['weather_dir'])  #TODO: remove this
        # self.path_dict['corres_data_path'] = weather_dir / country_code / 'results'

        data_dir = Path(config_obj.config['data_dir'])
        history_data_path = data_dir / country_code
        self.path_dict['data_file'] = history_data_path / (country_code + '_historical_inflow.csv')
        self.path_dict['history_data_path'] = history_data_path
        self.path_dict['cds_points_path'] = history_data_path / (country_code + '_' + self.geo_range+ config_obj.config['cds_source'] + '_points.xlsx')
        
        self.path_dict['up_basins_png_path'] = history_data_path / (country_code +'_' + self.geo_range + 'up_basins.png')
        self.path_dict['rivers_png_path'] = history_data_path / (country_code + '_rivers.png')
        self.path_dict['res_river_id_path'] = history_data_path / (country_code + '_res_river_id.csv')
        self.path_dict['lake_river_id_path'] = history_data_path / (country_code + '_lake_river_id.csv')
        self.path_dict['res_streamflow_path'] = history_data_path / (country_code + '_res_streamflow.csv')
        self.path_dict['lake_streamflow_path'] = history_data_path / (country_code + '_lake_streamflow.csv') 
        self.path_dict['logfile_hdam'] = history_data_path / (country_code + '_logfile_hdam.txt')
        self.path_dict['logfile_hror'] = history_data_path / (country_code + '_logfile_hror.txt')

        method = config_obj.config['algorithm']
        solution_dir = Path(config_obj.config['solution_dir'])
        self.path_dict['pred_data_path'] = solution_dir / str(method) / 'predicted_inflow'
        self.path_dict['his_eq_path'] = solution_dir / 'Historical_production'
        self.path_dict['his_price_path'] = solution_dir / 'price'
        self.path_dict['figs_path_hdam'] = solution_dir / str(method)  / 'figs' / 'hdam'
        self.path_dict['figs_path_hror'] = solution_dir / str(method)  / 'figs' / 'hror'
        
        self.path_dict['logfile_pred_hdam'] = self.path_dict['figs_path_hdam'] / (country_code + '_logfile_hdam.txt')
        self.path_dict['logfile_pred_hror'] = self.path_dict['figs_path_hror'] / (country_code + '_logfile_hror.txt')

        cds_source = config_obj.config['cds_source']
        cds_dir= Path(config_obj.config['cds_dir'])
        geo_grid_dir = Path(config_obj.config['geo_grid_dir'])
        self.path_dict['cds_dataset_path'] = cds_dir / cds_source
        self.path_dict['geo_grid_path'] = geo_grid_dir / (cds_source + '.shp')
        #Path(config_obj.config['cds_path'])

        # Generated directories by this tool
        self.__spine_gen_dir = ['history_data_path', 'pred_data_path', 'his_eq_path', 'his_price_path', 'figs_path_hdam', 'figs_path_hror', 'cds_dataset_path']

    def __create_dir(self):
        for dir in self.__spine_gen_dir:
            self.path_dict[dir].mkdir(parents=True, exist_ok=True)


