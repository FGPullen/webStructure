import numpy as np
from sklearn.manifold import TSNE
from page import Page
from pages import allPages
from sklearn.preprocessing import scale
import matplotlib.pyplot as Plot
from cluster import cluster

class visualizer:
	def __init__(self,pages):
		self.UP_pages = pages
		feature_matrix = []
		for page in self.UP_pages.pages:
			tfidf_vector = []
			for key in page.normtfidf:
				#tfidf_vector.append(page.normonehot[key])
				tfidf_vector.append(page.normtfidf[key])
				#tfidf_vector.append(page.Leung[key])
			feature_matrix.append(tfidf_vector)
		X = np.array(feature_matrix)
		# rescale X
		X = scale(X)
		model = TSNE(n_components=2, random_state=0)
		self.Y = model.fit_transform(X)

	def show(self,truth_list,pred_list,file_name,dataset):
		if dataset == "stackexchange":
			cluster_name = ["Others","Users","Questions","Index","Tags","Posts","Feeds"] # stackexchange
		elif dataset == "zhihu":	
			cluster_name = ["Users","Questions","Index","topic","collection","others"] # zhihu
		elif dataset == "rottentomatoes":
			cluster_name = ["celebrity","critics","top","m","trailers","guide","pictures"] #rotten
		elif dataset == "medhelp":
			cluster_name = ["groups","personal","forums","posts","tag","user"] # medhelp
		elif dataset == "asp":
			cluster_name = ["member","RedirectToLogin",'f','post','search',"others"]
		color_list = ["y","g","b","r","k","m","c","w"]
		marker_list = ["d","+","o","*","^","H","_","s"]
		label_count = [0 for i in range(len(marker_list))]
		x = self.Y[:,0]
		y = self.Y[:,1]
		print "Intotal we have " + str(x.size) + " data points"
		print str(len(truth_list)) + "\t" + str(x.size)
		assert len(truth_list) == x.size
		for i in range(len(truth_list)):
			m_index = truth_list[i]
			c_index = pred_list[i]
			mark =  color_list[c_index] + marker_list[m_index]
			Plot.plot(x[i],y[i],mark,label=cluster_name[m_index] if label_count[m_index]==0 else "")
			#Plot.plot(x[i],y[i],mark)
			label_count[m_index] = 1
		Plot.legend(numpoints=1,loc=3);
		#Plot.legend();
		self.write2D(file_name,pred_list)
		Plot.show()
	
	def write2D(self,file_name,group_list):
		write_file = open(file_name,"w")
		x = self.Y[:,0]
		y = self.Y[:,1]		
		assert len(self.UP_pages.pages) == x.size
		for i in range(x.size):
			write_file.write(self.filename2Url(self.UP_pages.pages[i].path)+"\t"+ str(group_list[i]) +"\t"+str(x[i])+"\t"+str(y[i])+"\n")
	
	def filename2Url(self,filename):
		return filename.replace("_","/")


if __name__=='__main__':
	#UP_pages = allPages(["../Crawler/toy_data/users/","../Crawler/toy_data/questions/","../Crawler/toy_data/lists/"])
	#UP_pages = allPages(["../Crawler/crawl_data/Users/","../Crawler/crawl_data/Outlinks_U/","../Crawler/crawl_data/Noise/"])
	UP_pages = allPages(["../Crawler/crawl_data/Questions/"])
	v = visualizer(UP_pages)
	user_group = cluster()
	for i in range(len(UP_pages.ground_truth)):
		if UP_pages.ground_truth[i] == 1:
			page = UP_pages.pages[i]
			user_group.addPage(page)
	global_threshold = len(UP_pages.pages) * 0.9
	print len(user_group.pages)
	user_group.find_local_stop_structure(UP_pages.nidf,global_threshold)

	v.show(v.UP_pages.ground_truth,"ground_truth.test")
	
	'''
	UP_pages = allPages(["../Crawler/crawl_data/Questions/"])
	feature_matrix = []
	for page in UP_pages.pages:
		tfidf_vector = []
		for key in page.tfidf:
			tfidf_vector.append(page.tfidf[key])
		feature_matrix.append(tfidf_vector)
	X = np.array(feature_matrix)
	X = scale(X)
	print len(UP_pages.pages)
	print X.shape
	model = TSNE(n_components=2, random_state=0)
	Y = model.fit_transform(X)

	x = v.Y[:,0]
	y = v.Y[:,1]
	print x.shape
	print y.shape
	Plot.plot(x[0:len(v.UP_pages.pages)],y[0:len(v.UP_pages.pages)], 'r*',label="questions")

	#Plot.plot(x[0:531],y[0:531], 'r*',label="users")
	#Plot.plot(x[531:],y[531:], 'g+',label="questions")

	#print Y[:,0].shape
	#print Y[:,1].shape
	#Plot.plot(Y[:,0],Y[:,1],'bo')
	#print UP_pages.pages[5].path
	#print UP_pages.pages[406].path
	#Plot.plot(x[0:531],y[0:531], 'r*',label="users")
	#Plot.plot(x[5],y[5], 'bo',label="102")
	#Plot.plot(x[406],y[406], 'yd',label="406")
	#Plot.plot(x[531:621],y[531:621], 'g+',label="outlinks")
	#Plot.plot(x[621:],y[621:], 'bo',label="noise")
	Plot.legend(loc=3);
	Plot.show()
'''
