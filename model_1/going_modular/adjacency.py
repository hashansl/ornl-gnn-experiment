# Import libraries
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import gudhi
from tqdm import tqdm
from persim import PersistenceImager
import invr
import matplotlib as mpl

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def generate_adjacent_counties(dataframe, variable_name, filtration):
    """Generate adjacent counties based on given dataframe and variable."""
    # currently using all of the data(without filtration)
    filtered_df = dataframe
    adjacent_counties = gpd.sjoin(filtered_df, filtered_df, predicate='intersects', how='left')
    adjacent_counties = adjacent_counties.query('sortedID_left != sortedID_right')
    adjacent_counties = adjacent_counties.groupby('sortedID_left')['sortedID_right'].apply(list).reset_index()
    adjacent_counties.rename(columns={'sortedID_left': 'county', 'sortedID_right': 'adjacent'}, inplace=True)
    adjacencies_list = adjacent_counties['adjacent'].tolist()
    county_list = adjacent_counties['county'].tolist()
    merged_df = pd.merge(adjacent_counties, dataframe, left_on='county', right_on='sortedID', how='left')
    merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')
    return adjacencies_list, merged_df, county_list

def form_simplicial_complex(adjacent_county_list, county_list):
    """Form a simplicial complex based on adjacent counties."""
    max_dimension = 3
    V = invr.incremental_vr([], adjacent_county_list, max_dimension, county_list)
    return V

def process_county(state, county_stcnty, variable,data_path,filtration=None):
    """Process data for a given state."""

    # return dictionary
    graph_data = {}

    svi_od_path = os.path.join(data_path, state, state + '.shp')
    svi_od = gpd.read_file(svi_od_path)
    # # for variable in selected_variables:
    #     # svi_od = svi_od[svi_od[variable] != -999]

    selected_variables_with_censusinfo = ['FIPS','gnn_id', 'STCNTY'] + [variable] + ['geometry']

    svi_od_filtered_state = svi_od[selected_variables_with_censusinfo].reset_index(drop=True)

    # Process the data for the given county
    
    # Filter the dataframe to include only the current county
    county_svi_df = svi_od_filtered_state[svi_od_filtered_state['STCNTY'] == county_stcnty]

    # Process the data for the given variable

    df_one_variable = county_svi_df[['STCNTY','FIPS','gnn_id', variable, 'geometry']]
    df_one_variable = df_one_variable.sort_values(by=variable)
    df_one_variable['sortedID'] = range(len(df_one_variable))
    df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
    df_one_variable.crs = "EPSG:3395"

    adjacencies_list, adjacent_counties_df, county_list = generate_adjacent_counties(df_one_variable, variable,filtration)
    adjacent_counties_dict = dict(zip(adjacent_counties_df['county'], adjacent_counties_df['adjacent']))
    county_list = adjacent_counties_df['county'].tolist()
    simplices = form_simplicial_complex(adjacent_counties_dict, county_list)

    graph_data['simplices'] = simplices
    graph_data['dataframe'] = df_one_variable

    return graph_data
