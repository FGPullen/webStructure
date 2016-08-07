from pages import allPages
from sklearn.preprocessing import normalize
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
'''
a = [3,3,3,2,2,1]
c = Counter(a)
x, y  =[], []
for item in c:
    x.append(item)
    y.append(c[item])
plt.plot(x,y)
plt.show()



'''
#pages = allPages(["../Crawler/test_data/zhihu/"],dataset="rottentomatoes",mode="raw")
pages = allPages(["../Crawler/Mar15_samples/asp/"],dataset="asp",mode="read")

tf_matrix = []
log_tf_matrix = []
for index, page in enumerate(pages.pages):
    if index == 1 or index == 989:
        print page.path
        vector = []
        for key in page.selected_tfidf:
            vector.append(page.selected_tfidf[key])
        tf_vector = normalize(vector,norm='l1')[0]
        tf_matrix.append(tf_vector)

        vector = []
        for key in page.selected_logtfidf:
            vector.append(page.selected_logtfidf[key])
        log_tf_vector = normalize(vector,norm='l1')[0]
        log_tf_matrix.append(log_tf_vector)
print tf_matrix
print log_tf_matrix
#ax = subplot(1,1,1)
plt.subplot(211)
line1 = plt.plot(tf_matrix[0],color="r",label="question_tf-idf")
line2= plt.plot(tf_matrix[1],color="g",label="user_tf-idf")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show()
'''
df = pages.selected_df
print pages.num
tmp = pages.df
df = tmp.values()
counter = Counter(df)
print counter

gold = pages.ground_truth
g_cnt = Counter(gold)
print g_cnt

x = []
y = []
df_counter = sorted(counter.iteritems(),key=lambda i:i[0],reverse=False)
for index,tuple in enumerate(df_counter):
    if tuple[0] > 3:
        x.append(tuple[0])
        y.append(tuple[1])
plt.plot(x,y)
plt.scatter(x,y,s=10)

axis = []
for key in g_cnt:
    axis.append(g_cnt[key])
    plt.plot([g_cnt[key],g_cnt[key]],[0,50],color="r")

plt.show()
'''