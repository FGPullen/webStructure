'''
This program will estimate the hits score from sitemap for each clustesr
@input1: the cluster results file with cluster id for each documents
@input2: the sampled document for each xpath of document (with corresponding cluster id)
@input3: the tf of each xpath for one page
@ouput: transition probabiltiy matrix for clusters
'''
import pickle
import random
import re
from lxml import etree
import os
import numpy as np

def read_trans_dict(dataset):
    path = "./site.trans/{}_trans.dict".format(dataset)
    dict_file = open(path,"r")
    dict = pickle.load(dict_file)
    return dict

def get_cluster_dict(dataset):
    path = "./Apr17/new_{0}.txt".format(dataset)
    lines = open(path,'r').readlines()
    cluster_dict = {}
    for line in lines:
        # this is a mistake...
        [page, gold, cluster] = line.strip().replace("../Crawler/Apr17/samples/{}/".format(dataset),"").split()
        cluster_id = int(cluster.replace("cluster:",""))
        cluster_dict[page] = cluster_id
        #print page, cluster_id
    return cluster_dict

def get_trans_mat(dataset,cluster_dict,trans_dict):
    folder_path = "../../Crawler/Apr17_samples/" + dataset
    cluster_num = len(set(cluster_dict.values()))
    print "cluster number is {}".format(cluster_num)
    trans_mat = np.zeros((cluster_num, cluster_num))
    count_list = {}
    print len(trans_dict)
    for key in trans_dict:
        trans = trans_dict[key]
        print key
        if key not in cluster_dict:
            continue
        cluster_id = cluster_dict[key]
        if cluster_id not in count_list:
            count_list[cluster_id] =1
        else:
            count_list[cluster_id] +=1
        print key
        for xpath,url_list in trans.iteritems():
            length = len(url_list)
            count = 0
            for url in url_list:
                if url in cluster_dict:
                    count += 1

            print "for xpath: {0}  --- {1} out of {2} have been crawled and have cluster id".format(xpath,count, length)
            if count == 0:
                continue
            ratio = float(length)/float(count)
            for url in url_list:
                if url in cluster_dict:
                    destination_id = cluster_dict[url]
                    print destination_id,
                    trans_mat[cluster_id][destination_id] += 1 * ratio
            print ratio

    print count_list
    print sum(count_list.values())
    for i in range(cluster_num):
        for j in range(cluster_num):
            if i not in count_list:
                trans_mat[i,j] = 0
            else:
                trans_mat[i, j] = float(trans_mat[i, j]) / float(count_list[i])
    #print trans_mat

    file = open("./Transition/{0}.mat".format(dataset), "w")
    for i in range(cluster_num):
        for j in range(cluster_num):
            if i == j:
                file.write(str(i) + " " + str(j) + " " + str(0) + "\n")
            else:
                file.write(str(i) + " " + str(j) + " " + str(trans_mat[i, j]) + "\n")
    return trans_mat

def transform(url,prefix):
    intra = intraJudge(url,dataset)
    if intra == 1:
        return prefix + url
    elif intra == 2:
        return url

def getAnchor(contents):
    link_dict = {}
    tree= etree.HTML(str(contents))
    Etree = etree.ElementTree(tree)
    nodes = tree.xpath("//a")
    for node in nodes:
        try:
            xpath = removeIndex(Etree.getpath(node))
            #print xpath,node.attrib['href']
            if xpath not in link_dict:
                link_dict[xpath] = []
            url = node.attrib['href']

            link_dict[xpath].append(url)
        except:
            err = "Oh no! " + str(node)
    #print "examine sampled link dict ", link_dict
    return link_dict

def removeIndex(xpath):
    indexes = re.findall(r"\[\d+\]",str(xpath))
    for index in indexes:
        xpath = xpath.replace(index,"")
    return xpath

def intraJudge(url, site):
    # oulink with http or symbol like # and /
        if site == "stackexchange":
            if url[0]=="/" and url[0:2] !="//":
                return 1
            else:
                if "http://android.stackexchange.com/" in url:
                    return 2
                else:
                    return 0
        elif site == "yelp":
            if len(url) == 1 or "http" in url:
                if "http://www.yelp.com" in url:
                    return 0
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                print url
                return 1
        elif site == "asp":
            if len(url) == 1 or "http" in url:
                if "http://forums.asp.net" in url:
                    return 0
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                print url
                return 1
        elif site == "douban":
            if "http" in url:
                if "movie.douban.com" in url:
                    return 2
                else:
                    return 0
            else:
                return 0
        elif site == "tripadvisor":
            if "http" in url:
                if "tripadvisor.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                return 1
        elif site == "hupu":
            if "http" in url:
                if "voice.hupu.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                return 1
        elif site == "biketo":
            if "http" in url:
                if "biketo.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//" and not url.endswith(".jpg"):
                return 1
            else:
                return 0
        elif site == "amazon":
            if "http" in url:
                return 0
            elif url[0:2] == "//":
                return 0
            else:
                return 1
        elif site == "youtube":
            if "https" in url:
                return 0
            elif url[0:2] == "//":
                return 0
            else:
                return 1
        elif site == "csdn":  # http://bbs.csdn.net/home # http://bbs.csdn.net
            if "http" in url:
                if "my.csdn.net" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                if "javascript:void(0)" in url:
                    return 0
                else:
                    return 1
        elif site == "baidu":
            if "http" in url:
                if "tieba.baidu.com" in url:
                    return 2
                else:
                    return 0
            elif url[0:2] == "//":
                return 0
            else:
                if "javascript:void(0)" in url:
                    return 0
                else:
                    return 1
        elif site == "huffingtonpost":
            if "http" in url:
                if "http://www.huffingtonpost.com/" in url and not url.endswith(".jpg"):
                    return 2
                else:
                    return 0
        else:
            return 0


if __name__ == "__main__":
    dataset = "stackexchange"
    trans_dict = read_trans_dict(dataset)
    cluster_dict = get_cluster_dict(dataset)
    get_trans_mat(dataset,cluster_dict,trans_dict)