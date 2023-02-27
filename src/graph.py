import json
import os
import pandas as pd
import networkx as nx


def load_graph_dataset(data_path="data/deezer_ego_nets"):
    with open(os.path.join(data_path, "deezer_edges.json")) as f:
        graph = nx.MultiGraph({
            node: [element[1] for element in neighbors]
            for node, neighbors in json.load(f).items()
        })
    target = pd.read_csv(os.path.join(data_path, "deezer_target.csv"))
    print(graph.edges, target)


