import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

scores1 = {
    ('track', 'genre_top'): 0.026445116715094422,
    ('track', 'language_code'): 0.0006954919225916563,
    ('album', 'type'): 0.024426920507276326
}
scores2 = {
    ('track', 'genre_top'): 0.016227047010185586,
    ('track', 'language_code'): 0.0005509310719005125,
    ('album', 'type'): 0.013834608265281307
}
scores3 = {
    ('track', 'genre_top'): 0.007059791187357992,
    ('track', 'language_code'): 0.00022645370113507854,
    ('album', 'type'): 0.0035551997726302716
}

df = pd.DataFrame([
    {"algo_name": "OneClassSVM", "external_var": "Genre Top", "MI": 0.026445116715094422},
    {"algo_name": "OneClassSVM", "external_var": "Language Code", "MI": 0.0006954919225916563},
    {"algo_name": "OneClassSVM", "external_var": "Album Type", "MI": 0.024426920507276326},

    {"algo_name": "IsolationForest", "external_var": "Genre Top", "MI": 0.016227047010185586},
    {"algo_name": "IsolationForest", "external_var": "Language Code", "MI": 0.0005509310719005125},
    {"algo_name": "IsolationForest", "external_var": "Album Type", "MI": 0.013834608265281307},

    {"algo_name": "DBSCAN", "external_var": "Genre Top", "MI": 0.007059791187357992},
    {"algo_name": "DBSCAN", "external_var": "Language Code", "MI": 0.00022645370113507854},
    {"algo_name": "DBSCAN", "external_var": "Album Type", "MI": 0.0035551997726302716},
])
g = sns.catplot(
    data=df, kind="bar",
    x="external_var", y="MI", hue="algo_name",
    errorbar="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("External Variable", "Mutual Information")
g.legend.set_title("Anomaly MI Per External Var")
plt.show()
#
# scores = {}
# for key in scores1.keys():
#     scores[key] = [scores1[key], scores2[key], scores3[key]])
#
# plt.bar(range(len(scores)), list(scores.values()), align='center')
# plt.xticks(range(len(scores)), list(scores.keys()))
# plt.show()
