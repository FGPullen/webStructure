
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
    #url_list = []
    counter = 0
    for line in open(path,"r").readlines():
        counter += 1
        if counter > num:
            break
        url = line.split(("\t")[0])
        id = int(line.split("\t")[-1])
        #if url not in url_list:
        id_list.append(id)
            #url_list.append(url)
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
        url = line.strip().split(" ")[0]
        url = url.replace("/","_")
        url_list.append(url)

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

def valid_ratio(counter,site):
    if site == "asp":
        invalid_list = [-1,2,4,11]
    elif site == "stackexchange":
        invalid_list = [-1,3,5,6,13,18,19]
    elif site == "youtube":
        invalid_list = [-1]
    elif site == "douban":
        invalid_list = [-1,4,26,27]
    elif site == "hupu":
        invalid_list = [-1]
    elif site == "rottentomatoes":
        invalid_list = [-1,5,11,13]

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

    #data_folder = "../../Crawler/full_data/{}".format(site)
    #result_file = "./results/bfs/{}_May1_0_bfs_size5000.txt".format(site)
    #write_path = "./{}_bfs_classify.txt".format(site)
    #classify_results_file(c,result_file,data_folder,write_path)

    #prefix = "http://android.{0}.com".format(site)
    #prefix = "http://forums.asp.net"
    #prefix = "https://www.youtube.com"

    #path = "./results/bfs/{}_July30_0_bfs_size10001.txt".format(site)
    #bfs = get_cluster_results(path,5000)
    #bfs = a.annotate_file(path)
    #bfs = get_annotation_cluster(path,a,num=1000)

    # general_crawl
    #g_path = "/bos/usr0/keyangx/webStructure/cluster/crawling/results/{}_May1_0_general_size3001.txt".format(site)
    #g_path = "./results/evaluate/general/{}_July30_1_general_size2001.txt".format(site)
    #g_list = get_crawl_results(g_path,2000)
    #g_list = get_annotation_cluster(g_path,a,num=1000)


    # random walk
    rw_path = "./results/evaluate/sampling/sampling_uniform_{0}_{1}_size5001.txt".format(site,date)
    rw_list = get_crawl_results(rw_path,num=1000)
    #rw_list = get_annotation_cluster(rw_path,a,num=5000)

    #random walk with pr correction
    #rwpc_path = "./results/evaluate/sampling/sampling_pagerank_{0}_{1}_size5001.txt".format(site,date)
    #rwpc_list = get_crawl_results(rwpc_path,num=5000)
    #rwpc_list = get_annotation_cluster(rwpc_path,a,num=5000)

    # visit ratio with the walk
    vr_path = "./results/evaluate/sampling/sampling_visit_ratio_{0}_{1}_size5001.txt".format(site,date)
    vr_list = get_crawl_results(vr_path,num=1000)
    #vr_list = get_annotation_cluster(vr_path,a,num=5000)
    # May1, snow-ball sampling
    #may1_path = "../May1/site.dbscan/{}.txt".format(site)
    #may_list = get_cluster_results(may1_path,num=1000)
    #may_list = get_annotation_cluster(may1_path,a,num=1000)
    #vr_list = may_list

    # random_sampling
    # June29
    June29_path = "../../Crawler/June29_samples/{}/".format(site)
    #june = get_classify_results(c, June29_path,site)
    baseline = "./src/data/{0}/random_sample.txt".format(site)
    june = get_crawl_results(baseline,num=1000)
    #june = get_annotation_cluster(baseline,a,num=5000)

    # random walk with visit ratio on cluster level
    rwcvr_path = "./results/evaluate/sampling/sampling_est_ratio_{0}_{1}_size1002.txt".format(site,date)

    rwcvr_list = get_crawl_results(rwcvr_path,num=1000)
    #rwepc_list = get_annotation_cluster(rwepc_path,a,num=5000)

    # random walk with estimated pagerank correction
    #rwepc_path = "./results/evaluate/sampling/sampling_est_pagerank_{0}_{1}_size1002.txt".format(site,date)
    rwepc_path = "./results/sampling/sampling_est_pagerank_{0}_{1}_size2001.txt".format(site,date)
    rwepc_list = get_crawl_results(rwepc_path,num=1000)

    # random walk with indegree correction
    #rw_indegree = "./results/sampling/random_indegree_{}_size1001.txt".format(site)
    #rw_indegree = get_crawl_results(rw_indegree)

    # random walk with cluster ratio correction
    #rw_ratio = "./results/sampling/random_est_prob_{}_size1001.txt".format(site)
    #rw_ratio = get_crawl_results(rw_ratio)




    print june
    print rw_list
    print rwepc_list
    #may_list = rescale_counter(june,may_list)
    #g_list = rescale_counter(june,g_list)
    plot_set = [june,rw_list]
    max_id = -1
    for list in plot_set:
        for key in list:
            if key > max_id:
                max_id = key

    #plt.axhline(y=1.0)
    #plot_line(g_list,max_id,"green",label="general crawling")
    #plot_line(bfs,max_id,"red",label="BFS")
    plot_line(rw_list,max_id,"blue",label="RW")
    #plot_line(rwpc_list,max_id,"magenta",label="RW with orcacle PR correction")
    plot_line(vr_list,max_id, "red",label="visit ratio")
    #plot_line(rw_list,max_id,"green",label="random walk")
    plot_line(june,max_id,"black",label="random sampling")
    plot_line(rwepc_list,max_id,"cyan",label="RW with predicted PageRank correction")
    #plot_line(rwcvr_list,max_id,"magenta",label="RW with cluster level ratio estimating")


    # bfs_crawl

    # compare general crawl , bfs_crawl, random_crawl, May1, random_sampling, target
    compare_list = [rw_list,vr_list,rwcvr_list, rwepc_list]
    #compare_list = [bfs,may_list,rwepc_list]
    compare_name = ["random walk", "visit ratio","RW by estimating visti ratio ","RW with predicted PageRank correction"]
    for index,list in enumerate(compare_list):
        rmse = RMSE(list,june,max_id)
        print rmse, compare_name[index]
    #D = c_may
    #plt.bar(range(len(D)), D.values(), align='center')
    #plt.xticks(range(len(D)), D.keys())

    plt.title(site)
    plt.legend()
    plt.show()


    print " == valid ratio == "
    for index, list in enumerate(compare_list):
        ratio = valid_ratio(list,site)
        print str(ratio), " {} ".format(compare_name[index])


    '''
    compare_methods(june,may_list)
    compare_methods(june,rwepc_list)
    compare_methods(june,rw_list)
    compare_methods(june,bfs)
    '''
