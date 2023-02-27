import os
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mutual_info_score


def load_dynamic_dataset(data_path="data/driftdataset"):
    dfs = []
    for filename in os.listdir(data_path):
        with open(os.path.join(data_path, filename), "r") as file:
            df_rows = []
            for line in file.readlines():
                curr_row = {}
                line = line.split(";")
                curr_row["gas_type"] = line[0]
                line = line[1].split(" ")
                curr_row["concentration"] = line[0]

                for sensor_value in line[1:]:
                    sensor_value = sensor_value.split(":")
                    if len(sensor_value) == 2:
                        curr_row[f"sensor_{sensor_value[0]}"] = float(sensor_value[1])
                df_rows.append(curr_row)
            dfs.append(pd.DataFrame(df_rows))
    return pd.concat(dfs, axis=0)


def main():
    df = load_dynamic_dataset()
    external_vars = ["gas_type", "concentration"]
    X = df.drop(external_vars, axis=1).values
    y = df[external_vars]

    a = [[[elem] for elem in row] for row in X]
    flat_list = [item for sublist in a for item in sublist]

    model = GaussianHMM(n_components=6, covariance_type="diag", n_iter=1000).fit(flat_list, lengths=X.shape[0])
    hidden_states = model.predict(flat_list)
    print(hidden_states)
    print("done")
    print(mutual_info_score(
        y["gas_type"].values,
        hidden_states
    ))


if __name__ == '__main__':
    main()
