import time
from sklearn.cluster import MiniBatchKMeans, KMeans 
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

test_data, _ = make_blobs(2000, n_features=2, cluster_std=2, centers=5)

km = KMeans(n_clusters=5)
mini_km = MiniBatchKMeans(n_clusters=5)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

for i, model in enumerate([km, mini_km]):
	t0 = time.time()
	model.fit(test_data)
	t1 = time.time()
	t = t1 - t0
	sse = model.inertia_
	axes[i].scatter(test_data[:, 0], test_data[:, 1], c=model.labels_)
	axes[i].set_xlabel('time: {:.4f} s'.format(t))
	axes[i].set_ylabel('SSE: {:.4f}'.format(sse))

axes[0].set_title("K-Means")
axes[1].set_title("Mini Batch K-Means")
plt.show()