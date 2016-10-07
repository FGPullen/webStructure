
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from urlparse import urlparse
from auto_crawler import crawler
from pageCluster import pageCluster
import collections
from url_annotator import annotator
import lxml.html
import matplotlib.pyplot as plt
import math


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

def plot_line(counter,max_id,color,label):
    total = float(sum(counter.values()))
    print total, " total"
    x,y = [],[]
    x.append(-1)
    y.append(float(counter[-1])/total)
    for key in range(max_id+1):
        if key==-1:
            continue
        x.append(key)
        y.append(float(counter[key])/total)
    plot = plt.plot(x,y,color=color,marker="+",label=label)
    return plot

def get_crawl_results(path,num=1000):
    id_list = []
    url_list = []
    counter = 0
    for line in open(path,"r").readlines():
        counter += 1
        if num > 3000:
            if counter < 3000:
                continue
        if counter > num:
            break
        url = line.split(("\t")[0])
        id = int(line.split("\t")[-1])
        #if url not in url_list:
        id_list.append(id)
        #    url_list.append(url)
    c = collections.Counter()
    c.update(id_list)
    return c

def get_cluster_results(path,num=1000):
    id_list = []
    counter = 0
    for line in open(path,"r").readlines():
        counter += 1
        if counter == 1:
            continue
        if counter > num+1:
            break
        if "\t" not in line:
            id = line.strip().split(" ")[-1]
        else:
            id = line.strip().split("\t")[-1]
        id = int(id.replace("cluster:",""))
        id_list.append(id)
    c = collections.Counter()
    c.update(id_list)
    return c

def get_classify_results(c,classify_path,site):
    write_file = open("./src/data/{}/random_sample.txt".format(site),"w")
    id_list = []
    for file in os.listdir(classify_path):
        file_path = classify_path  + file
        p, cluster_id = c.classify(file_path)
        id_list.append(cluster_id)
        print file, cluster_id
        write_file.write(file+"\t"+str(cluster_id)+"\n")
    c = collections.Counter()
    c.update(id_list)
    return c

# @input: crawler object, result_file (first col ), data_path for pages
def classify_results_file(c,result_file,data_path,write_path):
    write_file = open(write_path,"w")
    for line in open(result_file,"r").readlines():
        url = line.strip().split()[0]
        path = data_path + "/" + url.replace("/","_")
        if not path.endswith(".html"):
            path = path + ".html"
        #print path
        p, cluster_id = c.classify(path)
        write_file.write(url+"\t"+str(cluster_id)+"\n")


def rescale_counter(baseline,counter):
    for key in baseline:
        counter[key] = float(counter[key])/float(baseline[key])
    return counter


def get_annotation_cluster(path,annotator,num=1000):
    url_list = []
    counter = 0
    for line in open(path,"r").readlines():
        counter += 1
        if counter > num:
            break
        url = line.strip().split("\t")[0]
        url = url.replace("/","_")
        url_list.append(url)
    print url_list

    class_list = annotator.get_ground_truth(url_list)
    for index,value in enumerate(class_list):
        #print value, type(value)
        if value == -1:
            print url_list[index], "-1"

    c = collections.Counter()
    c.update(class_list)
    return c

# measuing the differences between two distribution
# input: list 1 and list 2
# output: RMSE
def RMSE(l1,l2,max_id):
    rmse = 0.0
    total_l1, total_l2 = 1.0 * sum(l1.values()), 1.0 * sum(l2.values())
    for i in range(0,max_id):
        rmse += (float(l1[i])/total_l1-float(l2[i])/total_l2)**2
    rmse = math.sqrt(rmse)
    return rmse

def valid_ratio(counter,site): # annotated by url pattern
    if site == "asp":
        invalid_list = [-1,0,3,7,9,12]
    elif site == "stackexchange":
        invalid_list = [-1,1,2,3,4,5,9,15,16,19]
    elif site == "youtube":
        invalid_list = [-1]
    elif site == "douban":
        invalid_list = [-1,4,26,27]
    elif site == "hupu":
        invalid_list = [-1]
    elif site == "rottentomatoes":
        invalid_list = [-1,0,9,20]

    total = sum(counter.values())
    num = 0
    for key in counter:
        if key in invalid_list:
            #print key, counter[key]
            num += counter[key]
    #print num, total
    valid_ratio = 1 - float(num)/float(total)
    #print valid_ratio, " valid ratio"

    return valid_ratio


def compare_methods(counter_a, counter_b):
    print counter_a.most_common()
    print counter_b.most_common()

if __name__ == '__main__':
    site = sys.argv[1]
    folder_path = "../../Crawler/July30_samples/{}/".format(site)
    date = "July30"
    #sitemap = pageCluster(site,date,folder_path,0)
    a = annotator(site)

    #c = crawler(site,date,None,None,eps=None,cluster_rank=0,crawl_size=None,rank_algo=None)


    path = "./results/bfs/{}_July30_0_bfs_size10001.txt".format(site)
    #bfs = a.annotate_file(path)
    bfs = get_annotation_cluster(path,a,num=2001)

    # random walk
    rw_path = "./results/evaluate/sampling/sampling_uniform_{0}_{1}_size3001.txt".format(site,date)
    rw_list = get_annotation_cluster(rw_path,a,num=2001)

    # general_crawl
    g_path = "./results/evaluate/general/{}_July30_1_general_size2001.txt".format(site)
    #g_path = "./results/evaluate/general/{}_July30_1_general_size2001.txt".format(site)
    #g_list = get_crawl_results(g_path,2000)
    g_list = get_annotation_cluster(g_path,a,num=2001)

    # structure+diversity
    sd_path = "./results/evaluate/general/structure_diversity/{}_July30_1_general_size1004.txt".format(site)
    sd_list = get_annotation_cluster(sd_path,a,num=1001)

    # info+diversity
    id_path = "./results/evaluate/general/info_diversity/{}_July30_1_general_size2002.txt".format(site)
    id_list = get_annotation_cluster(sd_path,a,num=1001)

    compare_list = [bfs,rw_list,g_list,sd_list,id_list]
    compare_name = ["bfs","RW","general crawl with 3 terms","general crawl with structure+diversity","info+diversity"]


    print " == valid ratio == "
    for index, list in enumerate(compare_list):
        ratio = valid_ratio(list,site)
        print str(ratio), " {} ".format(compare_name[index])

