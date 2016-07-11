import json
import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import sklearn.cluster as Cluster
import numpy as np
import matplotlib.pyplot as plt

def findEps(X):
    K = 100
    print type(X)
    print X.shape
    num_feat = X.shape[1]
    print num_feat, " number of features"
    bin_num = num_feat/5.0
    default_eps = 0.10
    kdist_list = []
    nbrs = NearestNeighbors(n_neighbors=K, algorithm="ball_tree").fit(X)
    distances, indices = nbrs.kneighbors(X)
    for dist in distances:
        #kdist_list += dist.tolist()[1:]
        kdist_list+= dist.tolist()[4:5]
    n, bins = np.histogram(kdist_list, bins=bin_num)
    print bin_num, "num_bins"
    n, bins, _ = plt.hist(kdist_list, bins=bin_num)
    plt.show()

    threshold = 4
    eps = default_eps
    for idx, val in enumerate(n):
        if idx > 5 and val < threshold:
            eps = bins[idx]
            break
    return eps


if __name__ == "__main__":
    #site = sys.argv[1]
    site = "rottentomatoes"
    path = "./data/{0}.feat.json".format(site)
    print path

    with open(path,"r") as outfile:
        feat_dict = json.load(outfile)

    print type(feat_dict)
    print len(feat_dict)


    xpath_set = set()
    measurement = []
    url_list = []
    for url in feat_dict.keys():
        print url
        url_list.append(url)
        diction = feat_dict[url]
        measurement.append(diction)

    vec = DictVectorizer()

    feat_array = vec.fit_transform(measurement).toarray()

    print feat_array

    transformer = TfidfTransformer(norm='l1',sublinear_tf=True)
    tfidf = transformer.fit_transform(feat_array)
    print tfidf[0]
    #t = tfidf[0].sum(axis=0)

    eps = findEps(tfidf)
    print eps
    db = Cluster.DBSCAN(eps=0.15, min_samples=4).fit(tfidf)
    train_y = db.labels_
    #for i in range(len(train_y)):
    #    print i, url_list[i], train_y[i]




