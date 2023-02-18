import json
import os
import ast
import pandas as pd
import networkx as nx


def load_static_dataset(data_path="data/fma_metadata"):
    features = pd.read_csv(os.path.join(data_path, "features.csv"), index_col=0, header=[0, 1, 2])
    tracks = pd.read_csv(os.path.join(data_path, "tracks.csv"), index_col=0, header=[0, 1])

    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
               ('track', 'genres'), ('track', 'genres_all')]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
               ('album', 'date_created'), ('album', 'date_released'),
               ('artist', 'date_created'), ('artist', 'active_year_begin'),
               ('artist', 'active_year_end')]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ('small', 'medium', 'large')
    try:
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            'category', categories=SUBSETS, ordered=True)
    except (ValueError, TypeError):
        # the categories and ordered arguments were removed in pandas 0.25
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            pd.CategoricalDtype(categories=SUBSETS, ordered=True))

    COLUMNS = [('track', 'genre_top'), ('track', 'license'),
               ('album', 'type'), ('album', 'information'),
               ('artist', 'bio')]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype('category')

    # tracks.loc ('track', 'genre_top')
    # features.loc 'mfcc'
    # do we need to scale the features?
    print(features)
    print(tracks)
    return features, tracks  # X, y


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


def load_graph_dataset(data_path="data/deezer_ego_nets"):
    with open(os.path.join(data_path, "deezer_edges.json")) as f:
        graph = nx.MultiGraph({
            node: [element[1] for element in neighbors]
            for node, neighbors in json.load(f).items()
        })
    target = pd.read_csv(os.path.join(data_path, "deezer_target.csv"))
    print(graph.edges, target)


