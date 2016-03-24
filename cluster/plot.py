from pages import allPages
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
pages = allPages(["../Crawler/test_data/zhihu/"],dataset="rottentomatoes",mode="raw")
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
    if tuple[0] > 10:
        x.append(tuple[0])
        y.append(tuple[1])
plt.plot(x,y)
plt.scatter(x,y,s=10)
plt.show()
'''
x = []
y = []
for key in counter:
    x.append(key)
    y.append(counter[key])
plt.plot(x,y)
plt.show()
'''
'''
N = 5
menMeans = (20, 35, 30, 35, 27)
menStd = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

womenMeans = (25, 32, 34, 20, 25)
womenStd = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, womenMeans, width, color='y', yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width)
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()
'''