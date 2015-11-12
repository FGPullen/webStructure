#coding=utf8 
import os
import time
import gensim, logging
from pages import allPages
from gensim.models import word2vec
from nltk.tokenize import WordPunctTokenizer
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#===============================================================================
# sentences = [['first', 'second','sentence'], ['second', 'sentence']]
# model = gensim.models.Word2Vec(sentences, min_count=1,workers=3)
# print(model.similarity('first','sentence'))
#===============================================================================
#sentences = word2vec.LineSentence('comment/comment_table_cleaned.txt')
#sentences = sentences.decode('latin-1').encode('utf8')
print("Program Starts")
sentences = []
UP_pages = allPages(["../Crawler/crawl_data/Questions/"])
for page in UP_pages.pages:
	sentences.append(page.dfs_xpaths_list)
print len(sentences)
model = gensim.models.Word2Vec(sentences,min_count=5,size=100,workers=5)
#print("The lengh of sentences is ")
#print(str(sentences.len()))
#model = gensim.models.Word2Vec.load('../model/MedHelp_tokenizer.model')
#model.train(sentences)
#b = model.most_similar(positive=['feminism'], topn=1)
#print(b)
model.save('./Data/word2vec.model')
#print(model['nurse'])
#print(model.most_similar(['nurse'],topn=3))
#print(model.most_similar(['agree'],topn=10))
#print(model.most_similar(['cancer'],topn=8))
#print(model.most_similar(positive=["pain", "disease"],topn=3))
