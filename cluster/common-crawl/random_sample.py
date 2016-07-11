import os
import sys,re
from urllib.parse import urlparse
import random
import collections
import lxml.html,lxml.etree
import warc,pickle
import traceback
import json

def getDFSXpaths(root, xpath_dict,xpath=""):
    for node in root:
        if type(node.tag) is not str:
            continue
        tag = node.tag
        new_xpath = "/".join([xpath, tag])
        if len(node) == 0:
            xpath_dict[new_xpath] += 1
        if len(node) != 0:
            getDFSXpaths(node, xpath_dict, new_xpath)

if __name__ == '__main__':
    site = sys.argv[1]
    site_dir = sys.argv[2]
    arc_file = os.path.join(site_dir, "{0}.arc.gz".format(site))
    #prefix = "http://www.{0}.com".format(site)

    f = warc.open(arc_file)
    record_num = 0
    for record in f:
        record_num += 1
    random_list = random.sample(range(record_num),1000)
    print (random_list)
    print (record_num)

    feature_dict = {}

    f = warc.open(arc_file)
    counter = 0
    for record in f:
        url = record['URL']
        counter += 1
        #url_dict[url] = record_num
        if counter not in random_list:
            continue
        contents = record.payload.read()

        try:
            xpath_dict = collections.defaultdict(int)
            root =lxml.etree.HTML(contents)
            getDFSXpaths(root,xpath_dict)
            feature_dict[url] = xpath_dict
        except:
            pass
            #traceback.print_exc()

    #with open("{}.feat.dict".format(site),"wb") as outfile:
    #    pickle.dump(feature_dict,outfile)

    #write_file = open("{}.feat.txt".format(site),"wb")
    with open("{}.feat.json".format(site),"w") as f:
        json.dump(feature_dict,f)


