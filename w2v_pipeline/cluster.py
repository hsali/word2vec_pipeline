import numpy as np
import h5py
import os, glob, itertools, collections

from clustering.similarity import load_embeddings, compute_document_similarity
from utils.os_utils import mkdir

from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE


def reorder_data(idx, X, S, labels):
    return X[idx], S[idx][:,idx], labels[idx]


if __name__ == "__main__":

    import simple_config
    

    config = simple_config.load("cluster")
    config_score = simple_config.load("score")

    output_dir = config["output_data_directory"]
    mkdir(output_dir)
    
    #W,WX = load_embeddings()

    n_clusters = int(config["n_clusters"])
    method = 'unique'


    f_sim = os.path.join(output_dir, config["f_similarity"])
    if not os.path.exists(f_sim):

        f_h5 = os.path.join(
            config_score["output_data_directory"],
            config_score["document_scores"]["f_db"],
        )
        
        h5_score = h5py.File(f_h5,'r')
        
        # Load the document scores
        print "Loading the document scores"
        X = np.vstack(h5_score[method][key] for key in h5_score[method])

        # Compute and save the similarity matrix
        print "Computing the similarity matrix"
        
        # Save the similarity matrix
        h5_sim = h5py.File(f_sim,'w')
        h5_sim[method] = compute_document_similarity(X)
        h5_sim.close()


    h5_sim = h5py.File(f_sim,'r')
    S = h5_sim[method]
    print S
    
    exit()
    
    # Cluster similarity matrix
    print "Clustering"
    clf = SpectralClustering(affinity="precomputed", n_clusters=n_clusters)
    clusters = clf.fit_predict(S)

    # Reorder the data so the data is the assigned cluster
    idx = np.argsort(clusters)
    X2, S2, labels = reorder_data(idx, X, S, clusters)
    print S2.shape

    # Plot the heatmap
    import seaborn as sns
    plt = sns.plt
    sns.heatmap(S2)
    sns.plt.show()
    exit()

    # Compute the tSNE
    local_embed = TSNE(n_components=2,verbose=1,
                       method='exact',
                       metric='precomputed')
    T = local_embed.fit_transform(1-S2)
    print T
    exit()

    

    print clusters


    exit()


    '''
    # Load document score data
    X0 = np.vstack([h5[method][name][:]
                    for name in input_names])
    n_docs, n_dims = X0.shape

    #RUHS = random_unit_hypersphere(n_dims)
    RSS = random_spectral_sampling(X0)

    Y0,Y1 = [],[]
    xplot = []
    for k in range(2,10,1):

        # Test this with some clusters?
        
        clf = skc.KMeans(n_clusters=k, n_jobs=-1)

        clusters0 = clf.fit_predict(X0)
        M = compute_cluster_measure(X0,clusters0)
        _,s,_ = np.linalg.svd(M)

        
        #s /= s.max()
        
        #s = s**2
        #f = s/s.sum()
        #inter_entropy = -(f*np.log(f)).sum()
        #print s.min()
        inter_entropy = s.min()

        s = np.diag(M)
        #s = (1-s)**2
        #f = s/s.sum()
        #intra_entropy = -(f*np.log(f)).sum()
        
        #intra_entropy = s.min()#-(f*np.log(f)).sum()
        intra_entropy = 0
        
        xplot.append(k)
        #Y0.append(intra_entropy-inter_entropy)
        Y0.append(inter_entropy)
        Y1.append(intra_entropy)
        print k,inter_entropy, intra_entropy
        continue
    
    import seaborn as sns
    sns.plt.plot(xplot,Y0,label="inter")
    sns.plt.plot(xplot,Y1,label="intra")
    
    sns.plt.legend()
    sns.plt.show()
    '''
    
    
    '''
        X1 = RSS(n_docs)
        clusters1 = clf.fit_predict(X1)
        C1 = compute_cluster_compactness(X1,clusters1)
        
        #for _ in range(10):
        #    clusters1 = clf.fit_predict(X1)
        #    C1 = compute_cluster_compactness(X1,clusters1)
        print k, C1.sum(), C0.sum()
        #gap = np.log(C1.sum()) - np.log(C0.sum())
        #print gap
        #exit()
    '''
    '''
        exit()

        # Load document score data
        X = np.vstack([h5[method][name][:]
                       for name in saved_input_names])

        # Load the categorical columns
        Y = []
        for name in saved_input_names:
            f_sql = os.path.join(pred_dir,name) + '.sqlite'
            engine = create_engine('sqlite:///'+f_sql)
            df = pd.read_sql_table("original",engine,
                                   columns=[column,])
            y = df[column].values
            Y.append(y)

        Y = np.hstack(Y)

        # Determine the baseline prediction
        y_counts = collections.Counter(Y).values()
        baseline_score = max(y_counts) / float(sum(y_counts))

        # Predict
        scores,errors,pred = categorical_predict(X,Y,method,config)

        text = "Predicting [{}] [{}] {:0.4f} ({:0.4f})"
        print text.format(method, column,
                          scores.mean(), baseline_score)

        PREDICTIONS[method] = pred
        ERROR_MATRIX[method] = errors

    # Build meta predictor
    META_X = np.hstack([PREDICTIONS[method] for method
                        in config["meta_methods"]])
    
    method = "meta"
    scores,errors,pred = categorical_predict(META_X,Y,method,config)

    text = "Predicting [{}] [{}] {:0.4f} ({:0.4f})"
    print text.format(method, column,
                      scores.mean(), baseline_score)

    PREDICTIONS[method] = pred
    ERROR_MATRIX[method] = errors
    '''
