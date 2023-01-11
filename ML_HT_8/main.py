import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("flats_for_clustering.tsv", sep='\t')

data = data.replace("parter", 0)
data = data.replace("niski parter", 0)
data = data.dropna()


for ind in data.index:
    if(data["Piętro"][ind] == "poddasze"):
        data["Piętro"][ind] = max(int(data["Liczba pięter w budynku"][ind]) - 1, 0)


data = StandardScaler().fit_transform(data)

pca = PCA(n_components=2)


data = pca.fit_transform(data)


kmeans = KMeans(n_clusters=5, n_init="auto").fit(data)

 
label = kmeans.predict(data)

u_labels = np.unique(label)
 
 
for i in u_labels:
    plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
plt.show()