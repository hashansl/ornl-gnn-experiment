# Import Libraries

import numpy as np
import pandas as pd
import adjacency
import os
import geopandas as gpd
from tqdm import tqdm

# Utility functions
def get_folders(location):
    """Get list of folders in a directory."""
    return [name for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))]

# Define the main function
if __name__ == "__main__":
    # Main execution

    # Load the data
    DATA_PATH = '/home/h6x/git_projects/data_processing/processed_data/SVI/SVI2018_MIN_MAX_SCALED_MISSING_REMOVED_GNN_ID'
    SAVE_PATH = '/home/h6x/git_projects/gnn/model_1/data/raw'
    VARIABLE = 'EP_POV'

    states = get_folders(DATA_PATH)

    for state in tqdm(states, desc="Processing states"):
        # print(f'Processing {state}')
        state_path = os.path.join(DATA_PATH, state)
        state_data = gpd.read_file(state_path)

        # Get the list of counties
        counties = state_data['STCNTY'].unique().tolist()

        for county in counties:
            # print(f'Processing {county}')
            graph_data = adjacency.process_county(state, county, VARIABLE, DATA_PATH)

            # save the graph data in a numpy file
            save_path = os.path.join(SAVE_PATH, county)
            np.save(save_path, graph_data)

    print('All states processed.')