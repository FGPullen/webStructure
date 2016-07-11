
import os
import sys
import collections
import lxml.html
import pickle
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize

def get_url_hash(site_dir):
    url_list = []
    url_dict = collections.defaultdict(int)
    for file in os.listdir(site_dir):
        url = file.replace("_","/").replace(".html","")
        if url not in url_list:
            url_list.append(url)

    for index, url in enumerate(url_list):
        url_dict[url] = index

    return url_list,url_dict

def parse_dir(site_dir,prefix,url_list,url_dict):

    inlink_dict = collections.defaultdict(set)

    for file in os.listdir(site_dir):
        inlink_set = set() # key:url , value: list of interger
        file_path = site_dir + file
        url = file.replace("_","/").replace(".html","")
        if url not in url_list:
            url_list.append(url)

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
                if not href.startswith('http'):
                    href = prefix + href
                href = href.replace("_","/")
                if href in url_dict.keys():
                    inlink_set.add(url_dict[href])
        except: 
            pass
        inlink_dict[url] = inlink_set

    return inlink_dict

def get_trans_mat(inlink_dict, url_dict,size):
    # get transition matrix sparse
    row,col,data = [],[],[]
    for url in inlink_dict:
        url_id = url_dict[url]
        #print url_id
        #print inlink_dict[url]
        for inlink_id in inlink_dict[url]:
            row.append(url_id)
            col.append(inlink_id)
            data.append(1.0)
    print len(row), " length"
    trans_mat = sps.csr_matrix((data,(row,col)), shape=(size,size))

    return trans_mat


def calc_pagerank(trans_mat,doc_num, alpha=0.8,max_iter=10):
    trans_mat = normalize(trans_mat, norm='l1', axis=1)
    pr_score = np.ones((doc_num,1))/float(doc_num)
    ite = 0
    while(ite < max_iter):
        print "iteration ", ite
        #previous_pr = pr_score
        pr_score = trans_mat.T.dot(pr_score)*alpha + (1-alpha)*np.ones((doc_num,1))/float(doc_num)
        ite += 1
        #change = np.sum(np.abs(pr_score-previous_pr))
    return pr_score



if __name__ == '__main__':
    site = sys.argv[1]
    site_dir = sys.argv[2]
    #site_entry = sys.argv[3]
    site_prefix = sys.argv[3]
    url_list, url_dict = get_url_hash(site_dir)
    num_pages = len(url_list)

    inlink_dict = parse_dir(site_dir, site_prefix, url_list, url_dict)
    trans_mat = get_trans_mat(inlink_dict,url_dict,num_pages)
    folder = "./data/new/{0}/".format(site)
    if not os.path.exists(folder):
        os.makedirs(folder)

    pr_scores = calc_pagerank(trans_mat,num_pages)



    # url_dict
    #print pr_scores
    with open("./data/new/{0}/{0}.pr_scores".format(site),"w") as outfile:
        pickle.dump(pr_scores,outfile,pickle.HIGHEST_PROTOCOL)

    pr_score_dict = collections.defaultdict(float)


    for index,value in enumerate(pr_scores):
        pr_score_dict[url_list[index]] = value[0]

    with open("./data/new/{0}/{0}.pr_dict".format(site),"w") as outfile:
        pickle.dump(pr_score_dict,outfile,pickle.HIGHEST_PROTOCOL)


    sorted_list = sorted(pr_score_dict.iteritems(),key=lambda k:k[1],reverse=True)
    print " pages with highest pagerank scores:"
    for i in range(10):
        print sorted_list[i][0]
        #, sorted_list[i][1]

    print " pages with lowest pagerank scores:"
    for i in range(10):
        print sorted_list[-i][0]
        #, sorted_list[-i][1]

    '''
    with open("./data/{0}/{0}.url_dict".format(site),"w") as outfile:
        pickle.dump(url_dict,outfile,pickle.HIGHEST_PROTOCOL)
    with open("./data/{0}/{0}.inlink_dict".format(site),"w") as outfile:
        pickle.dump(inlink_dict,outfile,pickle.HIGHEST_PROTOCOL)
    '''

