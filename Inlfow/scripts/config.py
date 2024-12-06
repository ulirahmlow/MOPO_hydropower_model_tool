import json
import os
from dotenv import load_dotenv

def get_config(config_name):
    with open(os.path.join(os.path.dirname(__file__), config_name), 'r') as f:
        return json.load(f)
    
load_dotenv()
config = get_config('_config.json')
country_code = config['bidding_zone']
aviability_zone = config['available_hdam_zone']+config['available_hror_zone']
assert country_code in aviability_zone, "Unsupported bidding zone code"
available_hdam_zone = config['available_hdam_zone']
available_hror_zone = config['available_hror_zone']
run_of_river_generation = config['run_of_river_generation']

geo_dir = os.getenv("geo_dir")
data_dir = os.getenv("data_dir")
solution_dir = os.getenv("solution_dir")
weather_dir = os.getenv("weather_dir")
entsoe_api_key = os.getenv("entsoe_api_token")
era5_grids_path=os.getenv("era5_grids_path")

method=config['algorithm']
level=config['level']


# external file
basin_filepath = os.path.join(geo_dir, "Hydrobasins\\hybas_eu_lev07_v1c.shp")
river_filepath = os.path.join(geo_dir, "Hydrorivers\\HydroRIVERS_v10_eu_shp\\HydroRIVERS_v10_eu.shp")
onshore_filepath = os.path.join(geo_dir, "onshore.geojson")
jrc_filepath = os.path.join(geo_dir, "jrc-hydro-power-plant-database.csv")
filtered_basin_filepath=os.path.join(geo_dir, f"basins_filtered\\{country_code}_basins_lev07")

corres_data_path=os.path.join(weather_dir, f"{country_code}\\results")


#internal file
#era5_grids_path=os.path.join(data_dir, "era5_points/era5_points.shp")
#country_map_path=os.path.join(data_dir, "country_mapping.xlsx")

his_data_path=os.path.join(data_dir, f"{country_code}")
corres_points_path=os.path.join(his_data_path, f"{country_code}_CorRES_points.xlsx")

logfile_hdam=os.path.join(his_data_path, f"{country_code}_logfile_hdam.txt")
logfile_hror=os.path.join(his_data_path, f"{country_code}_logfile_hror.txt")
pred_data_path=os.path.join(solution_dir, f"{method}\\predicted_inflow")
his_eq_path=os.path.join(solution_dir, "Historical_production")
his_price_path=os.path.join(solution_dir, "price")

figs_path_hdam=os.path.join(solution_dir, f"{method}\\figs\\hdam")
figs_path_hror=os.path.join(solution_dir, f"{method}\\figs\\hror")
logfile_pred_hdam=os.path.join(figs_path_hdam, f"{country_code}_logfile_hdam.txt")
logfile_pred_hror=os.path.join(figs_path_hror, f"{country_code}_logfile_hror.txt")

# check folder
if not os.path.exists(his_data_path):
    os.makedirs(his_data_path)

if not os.path.exists(pred_data_path):
    os.makedirs(pred_data_path)

if not os.path.exists(his_eq_path):
    os.makedirs(his_eq_path)

if not os.path.exists(his_price_path):
    os.makedirs(his_price_path)

if not os.path.exists(figs_path_hdam):
    os.makedirs(figs_path_hdam)

if not os.path.exists(figs_path_hror):
    os.makedirs(figs_path_hror)