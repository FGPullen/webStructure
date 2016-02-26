# WebStructure project @ CMU LTI
The idea is analyzing web pages' structures for a better crawler which extracts contents more intelligently.
## Crawler
This module crawls sites with annotated groups using simple BFS strategy.

## Cluster
This module extracts Xpaths and calculate features for clustering and classification.
HITS idea is also implemented in this module. 

* For clustering algorithm, the following six python files are most important:

**page.py:   
data structure for one web page

**pages.py:   
data structure for page collection of one website.


**kmeans.py wkmeans.py  

Those two take feature matrix as input and generate clusters as output.  
wkmeans means weighted kmeans. 



** pageCluster.py:  
Main function. The number of clusters are heuristically assigned in its main function.
To implement clustering, use command:
```python
python pageCluster.py dataset algo feature train(cv)
```

dataset is the parameter select from [zhihu,stackexchange,rottentomatoes,medhelp,asp]  
algo is the parameter select from [kmeans,wkmeans]  
features select from [tf-idf,log-tf-idf,binary]  
train(cv) select from [train,cv]  
* output: evluation metric and visulization for train  

** Batch file: cv_results.sh & train_results.sh  
Try all possible parameter settings and write results to files.  


** visualization.py  
Utilizing t-sne to reduce high-dimenstion vectors to two dimensions for visualization.  


## Xpath  
Test python libarary for xpath extraction. 
