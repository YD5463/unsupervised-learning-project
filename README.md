# unsupervised-learning-project

In the project we worked on 3 different datasets, each characterized by different data, dynamic,
graphic and static. Our project included finding the most optimal clustering algorithm for each dataset,
which we evaluated using internal and external methods, while optimizing the amount of dimensions,
also, we checked how the anomalies affect the quality of the clustering. We found that using silhouette
evaluation, Hierarchical clustering almost always produces the best results, but changes when evaluating
by external variables. Furthermore, we determine which external variable is most associated with clus-
ters. Finally, we visualized the static dataset in two steps, creating a qualitative visualization of the data
clusters


- overleaf url: https://www.overleaf.com/project/63dd1217871af3fdcd9c4f5c

## Datasets
1. Music Geners3- The data contain 106,574 music tracks. we extract per track multiple external variable
generes4, album type and language code5. - https://github.com/mdeff/fma
2. Deezer Ego Nets6- The data contain users collected from the music streaming service Deezer. The
data contain 9626 graphs each label with male or female. - https://snap.stanford.edu/data/deezer_ego_nets.html
3. Gas Sensor Dataset 7 - The data set contains 13,910 measurements from 16 chemical sensors exposed
to 6 gases at different concentration levels. The dataset was gathered during the period of 36 months.
We used the external Variables - https://archive.ics.uci.edu/ml/machine-learning-databases/00270/driftdataset.zip

## Discussions
In this project, we implemented a process consisting of three parts for three different datasets, finding anoma-
lies if there are any, amount of dimensions and choosing the type of clustering algorithm.
In the static dataset, when we used the silhoutte score, we obtained hierarchical clustering as the algorithm
that maximized the silhoutte, in addition, we saw in the measurements of the association of the external vari-
ables with the clustering algorithms that for hierarchical clustering the highest NMI is obtained for genere
type, also, most of the algorithms showed in the NMI that there is a high dependency between the clustering
algorithms and the genere type, which shows that in the coding of the songs there are elements related to the
genre, which is expected to happen. Also, we inspect a relationship between genere type and anomalies, this
is seen mainly in one class SVM where the NMI between the genere and label was high, as can be seen in
Figure 3.
In the dynamic dataset, we got that Hierarchical clustering and Birch are equal and are the best for clustering
according to the silhouette score, and we inspect that for them the concentration is the best associate exter-
nal variables, what that especially interesting, that with the exception of one algorithms, all the algorithms
are the best associate with the concentration, And not with the gas type which shows that there is a depen-
dency between the concentration and the clustering. Also, we saw a distinct relationship between genere
and anomalies, this is seen mainly in all the anomalies detection algorithms we ran, in all of them the NMI
between the concentrarion and label was high, as you can see in Figure 4.
In the dataset of the graphs we again accepted that Hierarchial clustering is the best clustering algorithm, on
the other hand when we evaluated according to gender we saw that GMM brings the best results, in addition
we saw that the representation of the graph greatly affects the results, finally the method we chose is to con-
vert the graph to Laplacian and extract the eigenvalues, while the that eigenvalues which we extract from the
Laplacian matrix increase the NMI between the clusters and the gender, the Louvain encoding and spectral
embedding which we try give lower result, we too inspect that using anomalies detection improve the results
of the hierarchical clustering as shown from Figure 2

