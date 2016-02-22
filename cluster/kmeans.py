# Author: Keyang Xu keyangx@andrew.cmu.edu
#K-Means Clustering #


import random as rd
import numpy as np
import collections
from scipy.sparse import csr_matrix

class KMeans(object):

    def __init__(self):
        self.name = "kmeans"

    def get_distance(self, data, centroid, distance):
        if distance == 'Euclidean':
            #print np.sum((data - centroid)**2,axis=1)
            return np.sum((data - centroid)**2,axis=1)
        elif distance == 'Cosine':
            similarity =  np.dot(data, centroid.T)
            row = similarity.shape[0]
            return np.ones(row) - np.dot(data, centroid.T)

    def get_center(self, data, distance):
        return data.sum(axis=0)/float((data.shape[0]))


    def dist_from_centers(self,indexes):
        t = len(indexes)
        #assign entities to cluster
        self.dist[:, t-1] = self.get_distance(self.data, self.data[indexes[t-1],:], self.distance)
        u = self.dist.argmin(axis=1)
        return self.dist[np.arange(self.dist.shape[0]), u]

    def choose_next_center(self,D):
        self.probs = D/D.sum()
        self.cumprobs = self.probs.cumsum()
        r = rd.random()
        index = np.where(self.cumprobs >= r)[0][0]
        return index

    def get_kpp_centroids(self,k):
        self.dist = np.ones([self.n_entities, k])
        indexes = [rd.randint(0,self.n_entities-1)]
        while len(indexes) < k:
            D = self.dist_from_centers(indexes)
            next_index = self.choose_next_center(D)
            indexes.append(next_index)
        return indexes


    def _k_means(self, data, k, init_centroids='kmeans++', max_ite=1000, distance='Euclidean'):
        #returns -1, -1, -1, -1, -1 if there is an empty cluster
        n_entities, n_features = data.shape[0],data.shape[1]
        self.n_entities = n_entities
        self.n_features = n_features
        if init_centroids=="random":
            centroids = data[rd.sample(range(n_entities), k), :]
        elif init_centroids == "kmeans++":
            indexes = self.get_kpp_centroids(k)
            centroids = data[indexes,:]
            #print "finish init_centroids"
        #print centroids
        previous_u = np.array([])
        previous_dist_total = 0.0
        ite = 1
        while ite <= max_ite:
            #print "========= " + str(ite) + "=========="
            dist = np.zeros([n_entities, k])
            #assign entities to cluster
            for k_i in range(k):
                dist[:, k_i] = self.get_distance(data, centroids[k_i, :], distance)
            #print dist_tmp
            u = dist.argmin(axis=1) # min distance with k centroids, return the index not the value
            #put the sum of distances to centroids in dist_tmp 
            dist_vector = dist[np.arange(dist.shape[0]),u]
            dist_total = np.sum(dist_vector)
            if np.array_equal(u, previous_u):
            #if abs(dist_total - previous_dist_total)<0.00001:
                return u, centroids, ite, dist_total
            #update centroids
            for k_i in range(k):
                entities_in_k = u == k_i # entities == k_i and u == k_i
                #Check if cluster k_i has lost all its entities
                if sum(entities_in_k) == 0:
                    index = dist_vector.argmax(axis=0)
                    u[index] = k_i
                    dist_vector[index] = 0.0
                    entities_in_k = index
                    #return np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])
                centroids[k_i, :] = self.get_center(data[entities_in_k, :], distance)
            previous_u = u[:]
            previous_dist_total = dist_total
            ite += 1
        return u, centroids, ite, dist_total
        # if out of number of max_iteration

    def k_means(self, data, k, init_centroids='kmeans++', distance='Euclidean', replicates=50, max_ite=100):
        #Weighted K-Means
        self.distance= distance
        self.data = data
        print distance
        final_dist = float("inf")
        avg_iteration = []
        f1_list = []
        for replication_i in range(replicates):
            #for i in range(max_ite):
                #print self._k_means(data, k, init_centroids, max_ite, distance)
            [u, centroids, ite, dist_tmp] = self._k_means(data, k, init_centroids, max_ite , distance)
            if u[0] == -1:
                continue
            #given a successful clustering, check if its the best
            if dist_tmp < final_dist:
                final_u = u[:]
                final_centroids = centroids[:]
                final_dist = dist_tmp
            print str(replication_i) + " has " + str(ite) + " iterations."
            avg_iteration.append(float(ite))


        final_ite = sum(avg_iteration)/float(len(avg_iteration))
        self.final_u, self.final_centroids, self.k = final_u, final_centroids, k
        return final_u, final_centroids, final_ite, final_dist

    #def init_kmeans_plus_plus(self, data, k):
    def k_means_classify(self, data, distance='Euclidean'):
        k, centroids = self.k, self.final_centroids
        n_entities, n_features = data.shape[0],data.shape[1]
        dist_tmp = np.zeros([n_entities, k])
        for k_i in range(k):
            dist_tmp[:, k_i] = self.get_distance(data, centroids[k_i, :], distance) # ** means exponent
        u = dist_tmp.argmin(axis=1)
        return u

if __name__=='__main__':
    t = KMeans()
    #data = csr_matrix([[1.0,0.0],[2.0,0.0],[0.0,3.0],[0.0,4.0]])
    data = np.array([[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.5,0.5]])
    #n_entities, n_features = data.shape[0],data.shape[1]
    #print n_entities
    #k = 2
    #centroids = data[rd.sample(range(n_entities), k), :]
    final_u, final_centroids, final_ite, final_dist = t.k_means(data,k=2, replicates=1,distance="Kmeans++")
    print final_u
    #print final_centroids
    #test_data = np.array([[0.0,4],[2.0,0.0]])
    #t.wk_means_classify(test_data)
