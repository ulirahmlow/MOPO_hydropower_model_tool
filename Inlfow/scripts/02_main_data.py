from reservoir_inflow import fetch_inflow
from ror_inflow import fetch_ror
from predict_inflow_run import inflow_prediction
from predict_ror import ror_P_prediction
import os
import config

country_code=config.country_code


def main():
    available_hdam_country=config.available_hdam_zone
    available_ror_country=config.available_hror_zone
    method=config.method
    level=config.level

    if country_code in available_hdam_country:
        print(f"{country_code} has availble reservoir inflow data online...")
        data_file=os.path.join(config.his_data_path, f"{country_code}_historical_inflow.csv")
        if not os.path.exists(data_file):
             print(f"Local historical inflow data does not exist. Fetching data from API...")
             fetch_inflow(country_code)
        else:
            print("Local historical inflow data already exists. Skipping data fetching.")
        
        print(f"Predicting inflow data for {country_code}...")
       
        if level == "bidding zone":
            if method == "linear_regression":
                inflow_prediction(country_code)
            else: 
                print("Not yet supported method for bidding zone!")
        else:
            print("Not yet supported level!")
        print("----------------Finished---------------")
        
    if country_code in available_ror_country:
        if config.run_of_river_generation:
            print(f"{country_code} has availble ror data online...")
            data_file = os.path.join(config.his_data_path,f"{country_code}_historical_ror_generation.csv")
            if not os.path.exists(data_file):
                print("Local historical ror generation data does not exist. Fetching data from API...")
                fetch_ror(country_code)
            else:
                print("Local historical inflow data already exists. Skipping data fetching.")
            
            print(f"Predicting ror data for {country_code}...")
            if level == "bidding zone":
                if method == "linear_regression":
                    ror_P_prediction(country_code)
                else: 
                    print("Not yet supported method for bidding zone!")
            else:
                print("Not yet supported level!")
            
            print("----------------Finished---------------")



if __name__ == "__main__":
    main()