
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from urlparse import urlparse
from auto_crawler import crawler
import collections
import lxml.html
import matplotlib.pyplot as plt


def get_length_counter(site,site_dir,prefix):
    inlink_dict = collections.defaultdict(set)
    for file in os.listdir(site_dir):
        inlink_set = set()
        file_path = site_dir + file
        #print file
        original = open(file_path,"r").read()
        contents = original.replace("\n","")
        try:
            root = lxml.html.fromstring(contents)
            hrefs = root.xpath('//a/@href')
            for href in hrefs:
                href_str = str(href).strip()
                if not href_str:
                    continue
                if href_str[0] == '#':
                    continue
                if href_str.startswith('javascript:'):
                    continue
                if href_str.startswith('mailto:'):
                    continue
                if href_str[0] == '/':
                    if len(href_str) > 1 and href_str[1] == '/':
                        continue
                    href_str = prefix + href_str

                if href.startswith('http'):
                    if not href.startswith(prefix):
                        continue
                #print "== ", href
                inlink_set.add(href)
                inlink_dict[file] = inlink_set

        except:
            raise
    length_list = []
    for key in inlink_dict:
        #print key, len(inlink_dict[key])
        length_list.append(len(inlink_dict[key]))
    c = collections.Counter()
    c.update(length_list)
    print c
    return c

def plot_line(counter,color,label):
    x,y = [],[]
    if -1 in counter:
        x.append(-1)
        y.append(counter[-1])
    for key,value in counter.iteritems():
        if key==-1:
            continue
        x.append(key)
        y.append(value)
    plot = plt.plot(x,y,color,label=label)
    return plot

def get_crawl_results(path,num=1000):
    id_list = []
    counter = 0
    for line in open(path,"r").readlines():
        counter += 1
        if counter > num:
            break
        id = int(line.split("\t")[-1])
        id_list.append(id)
    c = collections.Counter()
    c.update(id_list)
    return c

def get_cluster_results(path,num=1000):
    id_list = []
    counter = 0
    for line in open(path,"r").readlines():
        counter += 1
        if counter > num:
            break
        id = line.strip().split(" ")[-1]
        id = int(id.replace("cluster:",""))
        id_list.append(id)
    c = collections.Counter()
    c.update(id_list)
    return c


if __name__ == '__main__':
    site = sys.argv[1]
    folder_path = "../../../Crawler/May1_samples/{}/".format(site)
    date = "May1"

    #sitemap = pageCluster(site,date,folder_path,0)
    #c = crawler()


    prefix = "http://android.{0}.com".format(site)

    # general_crawl
    #g_path = "/bos/usr0/keyangx/webStructure/cluster/crawling/results/{}_May1_0_general_size3001.txt".format(site)
    g_path = "../results/{}_May1_0_general_size3001.txt".format(site)
    g_list = get_crawl_results(g_path)

    # random walk
    rs_path = "../results/sampling/random_{}_size985.txt".format(site)
    rs_list = get_crawl_results(rs_path)

    # May1
    may1_path = "../../May1/site.dbscan/{}.txt".format(site)
    may_list = get_cluster_results(may1_path,num=1000)

    # target
    #target_path = "../results/{}_May1_0_sort_size3001.txt".format(site)
    #t_list = get_crawl_results(target_path)

    # random_sampling



    print g_list
    print rs_list
    plot_line(g_list,"green",label="general crawling")
    plot_line(rs_list,"red",label="random walk")
    plot_line(may_list,"blue",label="snow-ball sampling")
    #plot_line(t_list,"black")
    # bfs_crawl


    # compare general crawl , bfs_crawl, random_crawl, May1, random_sampling, target

    #D = c_may

    #plt.bar(range(len(D)), D.values(), align='center')
    #plt.xticks(range(len(D)), D.keys())


    plt.legend()
    plt.show()