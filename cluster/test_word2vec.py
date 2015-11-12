import gensim
from pages import allPages
model = gensim.models.Word2Vec.load("./Data/word2Vec.model")
UP_pages = allPages(["../Crawler/crawl_data/Questions/"])
for page in UP_pages.pages:
	total = [0.0 for i in range(100)]
	n = 0
	for xpath in page.dfs_xpaths_list:
		try:
			total += model[xpath]
			n = n + 1
		except:
			n = n
	avg_embedding = []
	for i in range(len(total)):
		avg_embedding.append(total[i]/float(n))
	print avg_embedding
