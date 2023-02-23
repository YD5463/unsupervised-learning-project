import json
import os
import pandas as pd
import networkx as nx
from collections import defaultdict


def load_dynamic_dataset(data_path="data/driftdataset"):
    dfs = []
    for filename in os.listdir(data_path):
        with open(os.path.join(data_path, filename), "r") as file:
            df = pd.read_csv(
                file,
                sep="\s+",
                skiprows=1,
                usecols=[0, 7],
                names=['TIME', 'XGSM']
            )
            dfs.append(df)
    return pd.concat(dfs, axis=0)
    # time_series = defaultdict(list)
    # for i, row in tqdm(df.iterrows()):
    #     time_val = row["TIME"].split(";")
    #     XGSM_val = row["XGSM"].split(":")
    #     time_series[time_val[0]].append({
    #         "time": time_val[1],
    #         "gas_density": XGSM_val[1]
    #     })
    # print(sorted(list(time_series.keys())))


def load_graph_dataset(data_path="data/deezer_ego_nets"):
    with open(os.path.join(data_path, "deezer_edges.json")) as f:
        graph = nx.MultiGraph({
            node: [element[1] for element in neighbors]
            for node, neighbors in json.load(f).items()
        })
    target = pd.read_csv(os.path.join(data_path, "deezer_target.csv"))
    print(graph.edges, target)


