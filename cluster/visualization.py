import numpy as np
from sklearn.manifold import TSNE
from page import Page
from pages import allPages
import matplotlib.pyplot as Plot

if __name__=='__main__':
	UP_pages = allPages(["../Crawler/toy_data/users/","../Crawler/toy_data/questions/"])
	feature_matrix = []
	for page in UP_pages.pages:
		tfidf_vector = []
		for key in page.tfidf:
			tfidf_vector.append(page.tfidf[key])
		feature_matrix.append(tfidf_vector)
	X = np.array(feature_matrix)
	print X.shape
	model = TSNE(n_components=2, random_state=0)
	Y = model.fit_transform(X)

	x = Y[:,0]
	y = Y[:,1]
	print x.shape
	print x
	print y.shape
	#print Y[:,0].shape
	#print Y[:,1].shape
	#Plot.plot(Y[:,0],Y[:,1],'bo')
	Plot.plot(x[0:199],y[0:199], 'r*')
	Plot.plot(x[199:],y[199:], 'g+')
	
	Plot.show()
