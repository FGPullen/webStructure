import scipy.sparse as sps
import sys
from sklearn.preprocessing import normalize
import numpy as np


class HITS:
    def reader(self, path="./transition.txt"):
        self.transition_path = path
        file = open(self.transition_path, "r")
        row = []
        col = []
        data = []
        max_id = -1
        for line in file:
            [row_i, col_i, data_i] = line.strip().split()
            row_i = int(row_i)
            col_i = int(col_i)
            if row_i > max_id:
                max_id = row_i
            if col_i > max_id:
                max_id = col_i
            data_i = float(data_i)
            row.append(row_i)
            col.append(col_i)
            data.append(data_i)

        self.doc_num = max_id + 1
        print self.doc_num
        # self-contained

        trans_mat = sps.csr_matrix((data, (row, col)), shape=(max_id + 1, max_id + 1))
        # trans_mat = normalize(trans_mat, norm='l1', axis=1)
        self.trans_mat = trans_mat

        return trans_mat

    def __init__(self, path):
        self.reader(path)
        self.compute_scores()

    def compute_scores(self, max_iter=100):
        trans_mat = self.trans_mat  # need normalization
        print trans_mat
        # page rank score vector
        auth_score = np.ones((self.doc_num, 1)) / float(self.doc_num)
        hub_score = np.ones((self.doc_num, 1)) / float(self.doc_num)
        # previous_pr = np.zeros((doc_num,1))
        ite = 0
        while (ite < max_iter):
            # previous_pr = pr_score
            hub_score = trans_mat.dot(auth_score)
            auth_score = trans_mat.T.dot(hub_score)
            auth_score = normalize(auth_score, norm='l1', axis=0)
            hub_score = normalize(hub_score, norm='l1', axis=0)
            ite += 1
            # change = np.sum(np.abs(pr_score-previous_pr))
        self.auth_score = auth_score
        self.hub_score = hub_score
        print "authorative scores"
        print self.auth_score
        print "hub score "
        print self.hub_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["stackexchange", "douban", "youtube", "baidu", "tripadvisor"], help="dataset for computing transition matrix")
    args = parser.parse_args()
    dataset = args.dataset
    h_gold = HITS("./Transition_class/{0}.mat".format(args.dataset))
    h_estimate = HITS("./Transition/{0}.mat".format(args.dataset))

    gold_auth = h_gold.auth_score
    gold_hub = h_gold.hub_score
    estimate_auth = h_estimate.auth_score
    estimate_hub = h_estimate.hub_score
    transfer_auth = np.zeros(estimate_auth.shape)
    transfer_hub = np.zeros(estimate_hub.shape)
    transfer_count = np.zeros(estimate_auth.shape)
    path = "./Apr17/new_{0}.txt".format(dataset)
    lines = open(path,'r').readlines()
    cluster_dict = {}
    for line in lines:
        # this is a mistake...
        [page, gold, cluster] = line.strip().replace("../Crawler/Apr17/samples/{}/".format(dataset),"").split()
        cluster_id = int(cluster.replace("cluster:",""))
        class_id = int(gold.replace("gold:",""))
        if class_id !=-1 and cluster_id!=-1:
            transfer_auth[cluster_id,:] += gold_auth[class_id,:]
            transfer_hub[cluster_id,:] += gold_hub[class_id,:]
            transfer_count[cluster_id,:] += 1
    for key in range(transfer_hub.shape[0]):
        if transfer_count[key,:] !=0:
            transfer_hub[key,:] /= transfer_count[key,:]
            transfer_auth[key,:] /= transfer_count[key,:]
        #print page, cluster_id

    print "transfer autho"
    transfer_auth = normalize(transfer_auth, norm='l1', axis=0)
    print "transfer hub"
    transfer_hub = normalize(transfer_hub, norm='l1', axis=0)

    print transfer_hub
    print estimate_hub
    # kendall tau for hub
    right = 0
    count = 0
    for i in range(transfer_hub.shape[0]-1):
        for j in range(i+1,transfer_hub.shape[0]-1):
            if transfer_auth[i,:] ==0 or transfer_auth[j,:] == 0:
                continue
            count += 1
            t1 = transfer_auth[i,:] - transfer_auth[j,:]
            t2 = estimate_auth[i,:] - estimate_auth[j,:]

            if t1 ==0 and t2 == 0:
                right += 1
            elif t1 * t2 >0:
                right += 1
    print count , right
    print "kendall's tau is {}".format(float(2*right-count)/float(count))