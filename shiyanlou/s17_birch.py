import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import Birch

digits = datasets.load_digits()

# fig, axes = plt.subplots(1, 5, figsize=(12, 4))
# for i, image in enumerate(digits.images[:5]):
# 	axes[i].imshow(image, cmap=plt.cm.gray_r)
# plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(digits.data)
# print(digits.data.shape)
# print(pca_data)

# plt.figure(figsize=(10, 8))
# plt.scatter(pca_data[:,  0], pca_data[:, 1])

birch = Birch(n_clusters=10)
cluster_pca = birch.fit_predict(pca_data)
# print(cluster_pca)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_pca)
plt.show()
