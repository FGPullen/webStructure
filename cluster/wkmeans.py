# Author: Dr. Renato Cordeiro de Amorim, r.amorim@glyndwr.ac.uk
# Modifed by Keyang Xu
#Weighted K-Means - subspace clustering with cluster dependant weight
#
#data
#    numpy.array data set. It should be standardized (for best results). Format: entities x features
#k
#    Number of clusters.
#beta
#    The weight exponent, as per original paper
#init_centroids
#   Initial centroids. Format: centroids x features
#init_weights
#   Initial weights. Format: number of clusters x features
#dist
#   Distance in use.
#replicates
#    Number of times you want to run WK-Means. The method will return the clustering with smallest
#    WK-Means output criterion (weighted sum of distances between entities and respective centroids)
#p
#   The distance exponent of the Minkowski and Minkowski_pthPower distances. Not used if you select a different distance
#max_ite
#   The maximum number of iterations allowed in K-Means
########################################################################################################################
#
# More info:
#
#Huang, J.Z., Ng, M.K., Rong, H., Li, Z.: Automated variable weighting in k-means type clustering. Pattern Analysis and
## Machine Intelligence, IEEE Transactions on 27(5), 657-668 (2005)
#
#Huang, J.Z., Xu, J., Ng, M., Ye, Y.: Weighting method for feature selection in k-#means. Computational Methods of
#Feature Selection, Chapman & Hall/CRC pp. 193-209 (2008)


import random as rd
import numpy as np


class WKMeans(object):

    def __init__(self):
        self.name = "wkmeans"

    def get_distance(self, data , centroid, distance,weight=None):
        if distance == 'Euclidean':
            if weight is None:
                #print str(data) + "\t" + str(centroid) + "\t" + str(np.sum((data - centroid)**2))
                return np.sum((data - centroid)**2)

            else:
                #s = np.dot(((data - centroid)**2),np.ones(weight.reshape(-1,1).shape))
                s = np.dot(((data - centroid)**2), weight.reshape((-1,1)))
                return s[:,0]         



    def get_center(self, data, distance):
        return data.sum(axis=0)/float((data.shape[0]))


    def _get_centroid_based_weights(self,data, centroids, k , beta, u, n_features, distance):
        global_centroid = np.mean(data, axis=0)     
        dispersion = np.zeros([k, n_features])
        for k_i in range(k):
            for f_i in range(n_features):
                dispersion[k_i, f_i] = self.get_distance(centroids[k_i,f_i],global_centroid[f_i],distance)
        weights = np.zeros([k, n_features])                
        if beta != 1:
            exp = 1/(beta - 1)
            for k_i in range(k):
                for f_i in range(n_features):
                    weights[k_i, f_i] = (dispersion[k_i, f_i]**exp)/(dispersion[k_i, :]**exp).sum()
        else:
            for k_i in range(k):
                weights[k_i, dispersion[k_i, :].argmin()] = 1
        #print weights
        return weights

    def _get_dispersion_based_weights(self, data, centroids, k, beta, u, n_features, distance, dispersion_update='mean'):
        dispersion = np.zeros([k, n_features])
        for k_i in range(k):
            for f_i in range(n_features):
                dispersion[k_i, f_i] = self.get_distance(data[u == k_i, f_i], centroids[k_i, f_i], distance)
        '''
        print "----before---"
        print dispersion
        '''
        # smoothing using the global mean..
        if dispersion_update == 'mean':
            dispersion += dispersion.mean()
        else:
            dispersion += dispersion_update
        #calculate the actual weight
        '''
        print centroids
        print dispersion
        print u
        '''
        weights = np.zeros([k, n_features])
        if beta != 1:
            exp = 1/(beta - 1)
            for k_i in range(k):
                for f_i in range(n_features):
                    weights[k_i, f_i] = 1/((dispersion[k_i, f_i].repeat(n_features)/dispersion[k_i, :])**exp).sum()
        else:
            for k_i in range(k):
                weights[k_i, dispersion[k_i, :].argmin()] = 1
        #print weights
        return weights

    def __wk_means(self, data, k, beta, centroids=None, weights=None, weight_method=None, max_ite=100, distance='Euclidean'):
        #runs WK-Means (or MWK-Means) once
        #returns -1, -1, -1, -1, -1 if there is an empty cluster
        n_entities, n_features = data.shape[0],data.shape[1]
        if centroids is None:
            centroids = data[rd.sample(range(n_entities), k), :]
        if weights is None:
            if distance == 'Euclidean':
                #as per WK-Means paper
                weights = np.random.rand(k, n_features)
                weights = weights/weights.sum(axis=1).reshape([k, 1]) # normalize each line

        previous_u = np.array([])
        ite = 1
        while ite <= max_ite:
            #print "========= " + str(ite) + "=========="
            dist_tmp = np.zeros([n_entities, k])
            #assign entities to cluster
            for k_i in range(k):
                dist_tmp[:, k_i] = self.get_distance(data, centroids[k_i, :], distance, weights[k_i, :]**beta) # ** means exponent
            u = dist_tmp.argmin(axis=1) # min distance with k centroids, return the index not the value
            #put the sum of distances to centroids in dist_tmp 
            dist_tmp = np.sum(dist_tmp[np.arange(dist_tmp.shape[0]), u])
            #stop if there are no changes in the partitions
            if np.array_equal(u, previous_u):
                if weight_method is None:
                    weights = self._get_dispersion_based_weights(data, centroids, k, beta, u, n_features, distance)
                elif weight_method == "centroid":
                    print "centroids"
                    weights = self._get_centroid_based_weights(data,centroids,k, beta, u, n_features, distance)
                return u, centroids, weights, ite, dist_tmp
            #update centroids
            for k_i in range(k):
                entities_in_k = u == k_i # entities == k_i and u == k_i
                #Check if cluster k_i has lost all its entities
                if sum(entities_in_k) == 0:
                    return np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])
                centroids[k_i, :] = self.get_center(data[entities_in_k, :], distance)
            #update weights
            if weight_method is None:
                weights = self._get_dispersion_based_weights(data, centroids, k, beta, u, n_features, distance)
            elif weight_method == "centroid":
                print "centroids"
                weights = self._get_centroid_based_weights(data,centroids,k, beta, u, n_features, distance)
            previous_u = u[:]
            ite += 1

    def wk_means(self, data, k, beta, init_centroids=None, init_weights=None, weight_method=None, distance='Euclidean', replicates=100, max_ite=1000):
        #Weighted K-Means
        final_dist = float("inf")
        avg_iteration = []
        for replication_i in range(replicates):
            print replication_i
            #loops up to max_ite to try to get a successful clustering for this replication
            for i in range(max_ite):
                [u, centroids, weights, ite, dist_tmp] = self.__wk_means(data, k, beta, init_centroids, init_weights, weight_method, max_ite, distance)
                if u[0] != -1:
                    break
            if u[0] == -1:
                raise Exception('Cannot generate a single successful clustering')
            #given a successful clustering, check if its the best
            if dist_tmp < final_dist:
                final_u = u[:]
                final_centroids = centroids[:]
                final_dist = dist_tmp
                final_weights = weights
            print str(replication_i) + " has " + str(ite) + " iterations."
            avg_iteration.append(ite)
        final_ite = sum(avg_iteration)/float(len(avg_iteration))
        return final_u, final_centroids, final_weights, final_ite, final_dist

if __name__=='__main__':
    t = WKMeans()
    data = np.array([[2.0,0.0],[3.0,0.0],[0.0,3.0],[0.0,4.0]])
    final_u, final_centroids, weights, final_ite, final_dist = t.wk_means(data,2,2,weight_method="centroid", replicates=1)
    print final_u
    print final_centroids
    print weights
