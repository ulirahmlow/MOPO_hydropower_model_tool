#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from eu_hydro_inflow.user_argparse import UserArgparser
from eu_hydro_inflow.config_data_handle import ConfigData, FetchPath
from eu_hydro_inflow.fetch_inflow import FetchInflow
from eu_hydro_inflow.geoprocess import GeoBasins, GeoStreams
from eu_hydro_inflow.database_cds_api import DatabaseCds,Databasegeoglows

def main():
    arg_obj = UserArgparser()
    arg_obj.parser_run()
    config_obj = ConfigData()
    config_obj.args_check(arg_obj.args)
    path_obj = FetchPath(config_obj)
    inflow_obj = FetchInflow(config_obj, path_obj, arg_obj.args)
    geobasin_obj = GeoBasins(config_obj, path_obj)
    geobasin_obj.save_cds_grids()
    database_cds_obj = DatabaseCds(config_obj, path_obj)
    database_cds_obj.save_era5()
    geostream_obj = GeoStreams(config_obj, path_obj)
    geostream_obj.save_river_id()
    database_geoglows_obj = Databasegeoglows(config_obj, path_obj)
    database_geoglows_obj.save_streamflow()

if __name__ == "__main__":
    main()
    print("----------------Finished---------------")
