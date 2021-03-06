import numpy as np
import h5py
import os
import collections
from tqdm import tqdm

import simple_config
from sklearn.cluster import SpectralClustering

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import hierarchy

import utils.data_utils as uds


def subset_iterator(X, m, repeats=1):
    '''
    Iterates over array X in chunks of m, repeat number of times.
    Each time the order of the repeat is randomly generated.
    '''

    N, dim = X.shape
    progress = tqdm(total=repeats * int(N / m))

    for i in range(repeats):

        indices = np.random.permutation(N)

        for idx in np.array_split(indices, N // m):
            yield X[idx][:]
            progress.update()

    progress.close()


def cosine_affinity(X):
    epsilon = 1e-8
    S = cosine_similarity(X)
    S[S > 1] = 1.0  # Rounding problems
    S += 1 + epsilon

    # Sanity checks
    assert(not (S < 0).any())
    assert(not np.isnan(S).any())
    assert(not np.isinf(S).any())

    return S


def docv_centroid_order_idx(meta_clusters):
    dist = cdist(meta_clusters, meta_clusters, metric='cosine')

    # Compute the linkage and the order
    linkage = hierarchy.linkage(dist, method='average')
    d_idx = hierarchy.dendrogram(linkage, no_plot=True)["leaves"]

    return d_idx


class cluster_object(object):

    '''
    Helper class to represent all the constitute parts of a clustering
    '''

    def __init__(self):

        config = simple_config.load()["metacluster"]

        self.subcluster_m = int(config["subcluster_m"])
        self.subcluster_pcut = float(config["subcluster_pcut"])
        self.subcluster_repeats = int(config["subcluster_repeats"])
        self.subcluster_kn = int(config["subcluster_kn"])

        self.f_h5_centroids = os.path.join(
            config["output_data_directory"],
            config["f_centroids"],
        )

        score_method = config['score_method']
        DV = uds.load_document_vectors(score_method)
        self._ref = DV["_refs"]
        self.docv = DV["docv"]

        self.N, self.dim = self.docv.shape

    def compute_centroid_set(self):

        INPUT_ITR = subset_iterator(
            X=self.docv,
            m=self.subcluster_m,
            repeats=self.subcluster_repeats,
        )

        kn = self.subcluster_kn
        clf = SpectralClustering(
            n_clusters=kn,
            affinity="precomputed",
        )

        C = []

        for X in INPUT_ITR:
            # Remove any rows that have zero vectors
            bad_row_idx = ((X**2).sum(axis=1) == 0)

            X = X[~bad_row_idx]
            A = cosine_affinity(X)

            # "Force" symmetry due to rounding errors
            A = np.maximum( A, A.transpose() )

            labels = clf.fit_predict(A)

            # Compute the centroids
            (N, dim) = X.shape
            centroids = np.zeros((kn, dim))

            for i in range(kn):
                idx = labels == i
                mu = X[idx].mean(axis=0)
                mu /= np.linalg.norm(mu)
                centroids[i] = mu

            C.append(centroids)

        return np.vstack(C)

    def load_centroid_dataset(self, name):
        with h5py.File(self.f_h5_centroids, 'r') as h5:
            return h5[name][:]

    def compute_meta_centroid_set(self, C):
        print("Intermediate clusters", C.shape)

        # By eye, it looks like the top 60%-80% of the
        # remaining clusters are stable...

        nc = int(self.subcluster_pcut * self.subcluster_kn)
        clf = SpectralClustering(n_clusters=nc, affinity="precomputed")

        S = cosine_affinity(C)
        labels = clf.fit_predict(S)

        meta_clusters = []
        meta_cluster_size = []
        for i in range(labels.max() + 1):
            idx = labels == i
            mu = C[idx].mean(axis=0)
            mu /= np.linalg.norm(mu)
            meta_clusters.append(mu)
            meta_cluster_size.append(idx.sum())

        return np.array(meta_clusters)

    def compute_meta_labels(self, meta_clusters):

        n_clusters = meta_clusters.shape[0]

        msg = "Assigning {} labels over {} documents."
        print(msg.format(n_clusters, self.N))

        dist = cdist(self.docv, meta_clusters, metric='cosine')
        labels = np.argmin(dist, axis=1)

        print("Label distribution: ", collections.Counter(labels))
        return labels

    def docv_centroid_spread(self):
        meta_clusters = self.load_centroid_dataset("meta_centroids")
        meta_labels = self.load_centroid_dataset("meta_labels")
        n_clusters = meta_clusters.shape[0]

        mu, std, min = [], [], []
        for i in range(n_clusters):
            idx = meta_labels == i
            X = self.docv[idx]
            dot_prod = X.dot(meta_clusters[i])
            mu.append(dot_prod.mean())
            std.append(dot_prod.std())
            min.append(dot_prod.min())

        stats = np.array([mu, std, min])
        return stats


def metacluster_from_config(config):

    config = config['metacluster']
    os.system('mkdir -p {}'.format(config['output_data_directory']))

    CO = cluster_object()
    f_h5 = CO.f_h5_centroids

    # Remove the file if it exists and start fresh
    if os.path.exists(f_h5):
        os.remove(f_h5)

    h5 = uds.touch_h5(f_h5)

    # First compute the centroids
    C = CO.compute_centroid_set()

    # Now the meta centroids
    metaC = CO.compute_meta_centroid_set(C)

    # Find a better ordering for the centroids and reorder
    metaC = metaC[docv_centroid_order_idx(metaC)]

    h5["meta_centroids"] = metaC
    h5["meta_labels"] = CO.compute_meta_labels(metaC)
    h5["docv_centroid_spread"] = CO.docv_centroid_spread()

    h5.close()

if __name__ == "__main__":

    config = simple_config.load()
    metacluster_from_config(config)


#
#
# Old code below, useful for plotting the matrix?
#
#

'''
    def reorder(self,idx):
        self.X = self.X[idx]
        self.S = self.S[idx][:,idx]
        self.labels = self.labels[idx]
        self.T = self.T[idx]

    def __len__(self):
        return self.X.shape[0]

    def sort_labels(self):
        # Reorder the data so the data is the assigned cluster
        idx = np.argsort(self.labels)
        self.reorder(idx)

    def sort_intra(self):
        master_idx = np.arange(len(self))

        for i in self._label_iter:
            cidx = self.labels==i
            Z    = self.X[cidx]
            zmu  = Z.sum(axis=0)
            zmu /= np.linalg.norm(zmu)

            dist = Z.dot(zmu)
            dist_idx = np.argsort(dist)
            master_idx[cidx] = master_idx[cidx][dist_idx]

        self.reorder(master_idx)

    # Sort by labels first
    C.sort_labels()

    # Reorder the intra-clusters by closest to centroid
    C.sort_intra()

    # Plot the heatmap
    import pandas as pd
    import seaborn as sns
    plt = sns.plt

    fig = plt.figure(figsize=(9,9))
    print "Plotting tSNE"
    colors = sns.color_palette("hls", C.cluster_n)

    for i in C._label_iter:
        x,y = zip(*C.T[C.labels == i])
        label = 'cluster {}, {}'.format(i,C.words[i])

        plt.scatter(x,y,color=colors[i],label=label)
    plt.title("tSNE doc:{} plot".format(C.name_doc),fontsize=16)
    plt.legend(loc=1,fontsize=12)
    plt.tight_layout()


    fig = plt.figure(figsize=(12,12))
    print "Plotting heatmap"
    df = pd.DataFrame(C.S, index=C.labels,columns=C.labels)
    labeltick = int(len(df)/50)

    sns.heatmap(df,cbar=False,xticklabels=labeltick,yticklabels=labeltick)
    plt.tight_layout()
    #plt.savefig("clustering_{}.png".format(n_clusters))
    plt.show()
'''

'''
# Example of how to save strings in h5py

        if dtype in [str, unicode]:
            dt = h5py.special_dtype(vlen=unicode)
            h5.require_dataset(name, shape=result.shape, dtype=dt)
            for i, x in enumerate(result):
                h5[name][i] = x
'''
