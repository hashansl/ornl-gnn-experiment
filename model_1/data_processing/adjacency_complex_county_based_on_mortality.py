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
import io
from matplotlib.patches import Polygon
import pickle


from shapely.geometry import MultiPolygon, Polygon
from matplotlib.patches import Polygon as MplPolygon


# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Matplotlib default settings
mpl.rcParams.update(mpl.rcParamsDefault)

# Utility functions
def get_folders(location):
    """Get list of folders in a directory."""
    return [name for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))]

def generate_adjacent_counties(dataframe, variable_name,filtration): #adding filtration need to check correct or not here
    """Generate adjacent counties based on given dataframe and variable."""
    
    #get the 75th percentile of the variable
    low_v = dataframe[variable_name].quantile(filtration[0])
    high_v = dataframe[variable_name].quantile(filtration[1])

    # filter the dataframe based on the dataframe[variable_name] < high_v and dataframe[variable_name] > low_v at the same time

    filtered_df = dataframe[ (dataframe[variable_name] <= high_v) & (dataframe[variable_name] >= low_v)]

    # filtered_df = dataframe
    adjacent_counties = gpd.sjoin(filtered_df, filtered_df, predicate='intersects', how='left')
    adjacent_counties = adjacent_counties.query('sortedID_left != sortedID_right')
    adjacent_counties = adjacent_counties.groupby('sortedID_left')['sortedID_right'].apply(list).reset_index()
    adjacent_counties.rename(columns={'sortedID_left': 'county', 'sortedID_right': 'adjacent'}, inplace=True)
    adjacencies_list = adjacent_counties['adjacent'].tolist()
    county_list = adjacent_counties['county'].tolist()
    merged_df = pd.merge(adjacent_counties, dataframe, left_on='county', right_on='sortedID', how='left')
    merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')

    return adjacencies_list, merged_df, county_list,filtered_df

def form_simplicial_complex(adjacent_county_list, county_list):
    """Form a simplicial complex based on adjacent counties."""
    max_dimension = 3
    V = invr.incremental_vr([], adjacent_county_list, max_dimension, county_list)
    return V


def plot_simplicial_complex(dataframe, simplices, variable, base_path):
    # Extract city centroids
    city_coordinates = {row['sortedID']: np.array((row['geometry'].centroid.x, row['geometry'].centroid.y)) for _, row in dataframe.iterrows()}

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_axis_off() 

    # Plot the "dataframe" DataFrame
    dataframe.plot(ax=ax, edgecolor='black', linewidth=0.3, color="white")

    # # Plot the centroid of the large square with values
    # for _, row in dataframe.iterrows():
    #     centroid = row['geometry'].centroid
    #     text_to_display = f"{row[variable]:.3f}"
    #     plt.text(centroid.x, centroid.y, text_to_display, fontsize=10, ha='center', color="black")

    # Plot edges and triangles from simplices
    for edge_or_triangle in simplices:
        # Color sub-regions based on how it enters the simplicial complex in adjacency method
        if len(edge_or_triangle) == 1:
            vertex = edge_or_triangle[0]
            geometry = dataframe.loc[dataframe['sortedID'] == vertex, 'geometry'].values[0]

            # Handle both Polygon and MultiPolygon
            if isinstance(geometry, Polygon):
                ax.add_patch(MplPolygon(np.array(geometry.exterior.coords), closed=True, color='orange', alpha=0.3))
            elif isinstance(geometry, MultiPolygon):
                # Iterate through the individual polygons in the MultiPolygon
                for poly in geometry.geoms:
                    ax.add_patch(MplPolygon(np.array(poly.exterior.coords), closed=True, color='orange', alpha=0.3))

        elif len(edge_or_triangle) == 2:
            # Plot an edge
            ax.plot(*zip(*[city_coordinates[vertex] for vertex in edge_or_triangle]), color='red', linewidth= 1)

        elif len(edge_or_triangle) == 3:
            # Plot a triangle
            ax.add_patch(plt.Polygon([city_coordinates[vertex] for vertex in edge_or_triangle], color='green', alpha=0.2))

    # Save the plot
    plt.savefig(f'{base_path}/{variable}.png', dpi=300)
    plt.close(fig)

def process_entire_us(dataframe, selected_yr_names,filtration,base_path):
    """Process data for the entire US."""

    # return dictionary
    graph_data = {}

    selected_year_var_name = selected_yr_names[0]

    df_one_year = dataframe[['FIPS', selected_year_var_name, 'geometry']] #STCNCY CHANGED TO FIPS
    df_one_year = df_one_year.sort_values(by=selected_year_var_name)
    df_one_year['sortedID'] = range(len(df_one_year))
    df_one_year = gpd.GeoDataFrame(df_one_year, geometry='geometry')
    df_one_year.crs = "EPSG:3395"

    adjacencies_list, adjacent_counties_df, county_list,filtered_df = generate_adjacent_counties(df_one_year, selected_year_var_name,filtration) # Ishere correct to add filtration
    adjacent_counties_dict = dict(zip(adjacent_counties_df['county'], adjacent_counties_df['adjacent']))
    county_list = adjacent_counties_df['county'].tolist()
    simplices = form_simplicial_complex(adjacent_counties_dict, county_list)

    graph_data['simplices'] = simplices
    graph_data['dataframe'] = filtered_df

    # print(simplices)

    plot_simplicial_complex(df_one_year, simplices, selected_year_var_name,base_path)

    return graph_data


# Define the main function
if __name__ == "__main__":
    # Main execution
    base_path = '/home/h6x/git_projects/ornl-gnn-experiment/model_1/data/processed/GNN_simplices_data_2018'
    data_path = '/home/h6x/git_projects/ornl-gnn-experiment/model_1/data/processed/2018/svi_od_ranked_2018.shp'
    plot_path = '/home/h6x/git_projects/gnn/model_1/plots'


    # load the data
    overdose_df = gpd.read_file(data_path)

    # required columns
    required_columns = ['ST', 'STATE','COUNTY','FIPS','GEO ID','NOD_2018', 'geometry','label_90','label_80','label_95','GNN_ID']

    node_features = [
        'EP_POV', 'EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
        'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
        ]
    
    required_columns = required_columns + node_features

    #filter by STATE
    # overdose_df = overdose_df[overdose_df['STATE'] == 'FLORIDA']

    print(f'shape of the dataframe: {overdose_df.shape}')

    # filter the columns
    overdose_df = overdose_df[required_columns]

    # drop the ST rows with 'DC' # not dropping because generating maps for the entire US
    # overdose_df = overdose_df[overdose_df['STATE'] != 'DISTRICT OF COLUMBIA']

    states = overdose_df['STATE'].unique().tolist()

    print(f'Number of states: {len(states)}')

    print("Processing the entire US")

    selected_variables = ['NOD_2018']

    # selected_variablesa_ = [['NOD_2014'], ['NOD_2015'], ['NOD_2016'], ['NOD_2017'], ['NOD_2019'], ['NOD_2020']

    # for selected_variables in selected_variablesa_:

    #min max scaled the selected variables
    overdose_df[selected_variables] = (overdose_df[selected_variables] - overdose_df[selected_variables].min()) / (overdose_df[selected_variables].max() - overdose_df[selected_variables].min())

    # PRINT THE MAX AND MIN VALUES OF THE SELECTED VARIABLES
    for var in selected_variables:
        print(f'{var} max: {overdose_df[var].max()}')
        print(f'{var} min: {overdose_df[var].min()}')

    graph_data = process_entire_us(overdose_df, selected_variables,[0.0,1],plot_path)

    print(graph_data['dataframe'].shape)

    # save df as csv file
    graph_data['dataframe'].to_csv(f'{base_path}/processed_filtered_df_us.csv', index=False)

    # save simplices to a numpy file
    # np.save(f'{base_path}/simplices.npy', graph_data['simplices'],allow_pickle=True)
    with open(f'{base_path}/simplices_us.pkl', 'wb') as f:
        pickle.dump(graph_data['simplices'], f)

    

    print('processing done.')


    # simplices can be changed to FIPS - this coud help to keep the index same - right now sortrd ID is used
    # check the adjacency function - it should be correct
    # check the filtration - it should be correct



