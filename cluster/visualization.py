import numpy as np
from sklearn.manifold import TSNE
from page import Page
from pages import allPages
from sklearn.preprocessing import scale
import matplotlib.pyplot as Plot

if __name__=='__main__':
	UP_pages = allPages(["../Crawler/toy_data/users/","../Crawler/toy_data/questions/","../Crawler/toy_data/articles/"])
	feature_matrix = []
	for page in UP_pages.pages:
		tfidf_vector = []
		for key in page.tfidf:
			tfidf_vector.append(page.tfidf[key])
		feature_matrix.append(tfidf_vector)
	X = np.array(feature_matrix)
	#X = scale(X)
	print X.shape
	model = TSNE(n_components=2, random_state=1)
	Y = model.fit_transform(X)

	x = Y[:,0]
	y = Y[:,1]
	print x.shape
	print x
	print y.shape
	#print Y[:,0].shape
	#print Y[:,1].shape
	#Plot.plot(Y[:,0],Y[:,1],'bo')
	Plot.plot(x[0:200],y[0:200], 'r*')
	Plot.plot(x[200:400],y[200:400], 'g+')
	Plot.plot(x[400:],y[400:], 'bo')
	
	Plot.show()
