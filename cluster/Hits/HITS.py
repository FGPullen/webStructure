import scipy.sparse as sps
import sys
from sklearn.preprocessing import normalize
import numpy as np

class HITS:

	def reader(self, path="./transition.txt"):
		self.transition_path = path
		file = open(self.transition_path,"r")
		row = []
		col = []
		data = []
		max_id = -1
		for line in file:
			[row_i,col_i,data_i] = line.strip().split()
			row_i = int(row_i)
			col_i = int(col_i)
			if row_i > max_id:
				max_id = row_i
			if col_i > max_id:
				max_id = col_i
			data_i = float(data_i)
			row.append(row_i-1)
			col.append(col_i-1)
			data.append(data_i)

		self.doc_num = max_id
		print self.doc_num
		# self-contained
		for i in range(max_id):
			row.append(i)
			col.append(i)
			data.append(1)

		trans_mat = sps.csr_matrix((data,(row,col)), shape=(max_id,max_id))
		trans_mat = normalize(trans_mat, norm='l1', axis=1)
		self.trans_mat = trans_mat

		return trans_mat


	def __init__(self,path):
		self.reader(path)
		self.compute_scores()


	def compute_scores(self, max_iter=50):
		trans_mat = self.trans_mat # need normalization
		# page rank score vector
		auth_score = np.ones((self.doc_num,1))/float(self.doc_num)
		hub_score = np.ones((self.doc_num,1))/float(self.doc_num)
		#previous_pr = np.zeros((doc_num,1))
		ite = 0
		while(ite < max_iter):
			#previous_pr = pr_score
			hub_score = trans_mat.dot(auth_score)
			auth_score = trans_mat.T.dot(hub_score)
			auth_score = normalize(auth_score,norm='l1',axis=0)
			hub_score = normalize(hub_score,norm='l1',axis=0)
			ite += 1
			#change = np.sum(np.abs(pr_score-previous_pr))
		self.auth_score = auth_score
		self.hub_score = hub_score
		print "authorative scores"
		print self.auth_score
		print "hub score "
		print self.hub_score

if __name__ == "__main__":
	h = HITS("./Hits/stackexchange_mat.txt")
