
# Import Libraries
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, download_url, Data
import numpy as np 
import os
from tqdm import tqdm
import os.path as osp
import geopandas as gpd 
# import adjacency
from torch_geometric.utils import remove_self_loops



class OpioidDataset(Dataset):
    #ok
    def __init__(self, root,test=False, transform=None, pre_transform=None, pre_filter=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        # self.filename = filename
        super(OpioidDataset, self).__init__(root, transform, pre_transform)
    #ok
    @property
    def raw_file_names(self):
        """ If this file exists in raw directory, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return ['processed_filtered_df.csv','svi_od_ranked_2018.csv','simplices.pkl']
    #ok
    @property
    def processed_file_names(self):
        """ If these files are found in processed directory, processing is skipped"""
        if self.test:
            return [f'data_test.pt']
        else:
            return [f'data.pt']
    #ok
    def download(self):
        pass

    def flatten_and_unique(self,nested_list):
        # Flatten the nested list using a list comprehension
        flat_list = [item for sublist in nested_list for item in sublist]
        # Use a set to get unique elements and convert it back to a list
        unique_list = list(set(flat_list))
        return unique_list
    
    def flatten(self,nested_list):
        # Flatten the nested list using a list comprehension
        flat_list = [item for sublist in nested_list for item in sublist]
        return flat_list
    
    def _get_node_features(self, svi_df,filtered_df,var_names, node_ids):
        """ This will return a matrix / 2d array of the shape
        [Number of Nodes, Node feature size]

        In here is sorted ids is used as the node ids
        """
        
        #filtered_df should sort by sortedID
        filtered_df = filtered_df.sort_values(by=['sortedID'])   

        # get the FIPS to a list
        fips = filtered_df['FIPS'].tolist()

        filtered_svi = svi_df[svi_df['FIPS'].isin(fips)]
        filtered_svi.set_index('FIPS', inplace=True)

        # add the sortedID to the filtered_svi by matching the FIPS with the filtered_df
        filtered_svi = filtered_svi.merge(filtered_df[['FIPS', 'sortedID']], on='FIPS', how='left')
        filtered_svi.set_index('sortedID', inplace=True)
        # sort the df by sortedID
        filtered_svi = filtered_svi.sort_index()

        # filter the columns to get only the node features
        attributes = filtered_svi[var_names]

        return torch.tensor(attributes.to_numpy(), dtype=torch.float),filtered_svi

        
    def process(self):
        # print("This should not be running")


        NODE_FEATURES = [
        'EP_POV', 'EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
        'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
        ]

        # Load the data
        filtered_df = pd.read_csv(self.raw_paths[0])
        svi_od_ranked = pd.read_csv(self.raw_paths[1])
        simplices = pd.read_pickle(self.raw_paths[2])

        # Edge list - simplices(sortedID as node id)
        edges = []
  
        for i in range(len(simplices)):

            if len(simplices[i]) == 2:
                edges.append(simplices[i])
                edges.append(simplices[i][::-1]) # Reverse the order of the nodes because it is an undirected graph (1,0) and (0,1) are both needed for the edge list make it undirected

        #  Edge index --- COO format for PyG
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Node features
        node_ids  = self.flatten_and_unique(edges)

        # Get node features for entire graph
        attrs,filtered_svi = self._get_node_features(svi_od_ranked,filtered_df,NODE_FEATURES,node_ids)   #fips_sorted is the FIPS sorted by sortedID

        # Get labels for the nodes
        labels = filtered_svi['label_90'].tolist()   # filtered svi is sorted by sortedID so the labels are in the same order as the node_ids -- labels should look like [0,0,0,....1,1,1] because it sorted by mortality rate

        labels = torch.tensor(labels, dtype=torch.long)  # Use long for classification

        # Create the graph data object
        graph = Data(x=attrs, edge_index=edge_index, y=labels)

        # if self.pre_filter is not None and not self.pre_filter(graph):
        #     continue

        if self.pre_transform is not None:
            graph = self.pre_transform(graph)

        torch.save(graph, osp.join(self.processed_dir, f'data.pt'))


    def _get_labels(self, svi_od_ranked, node_ids):
        """
        This function matches the labels from the svi_od_ranked dataframe to the node IDs.
        """
        pass

    def len(self):
        # return self.data.shape[0]
        return 1

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
            
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data.pt'))        
        return data
    


# # Define the main function
# if __name__ == "__main__":
#     # Main execution

#     root_name = "/home/h6x/git_projects/ornl-gnn-experiment/model_1/going_modular/data"

#     dataset =OpioidDataset(root_name, test=False)

#     print()
#     print(f'Dataset: {dataset}:')
#     print('====================')
#     print(f'Number of graphs: {len(dataset)}')
#     print(f'Number of features: {dataset.num_features}')
#     print(f'Number of classes: {dataset.num_classes}')

#     data = dataset[0]  # Get the first graph object.

#     print()
#     print(data)
#     print('=============================================================')

#     # Gather some statistics about the first graph.
#     print(f'Number of nodes: {data.num_nodes}')
#     print(f'Number of edges: {data.num_edges}')
#     print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
#     print(f'Has isolated nodes: {data.has_isolated_nodes()}')
#     print(f'Has self-loops: {data.has_self_loops()}')
#     print(f'Is undirected: {data.is_undirected()}')




