from hits_estimate import read_trans_dict,get_cluster_dict
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from pageCluster import pageCluster
from pages import allPages
from page import Page
import math

class crawler:

    def __init__(self,dataset,entry,prefix):
        self.dataset = dataset
        self.folder_path = ["../../Crawler/Apr17_samples/{}/".format(dataset.replace("new_",""))]
        self.sitemap = pageCluster(dataset,self.folder_path,0)
        self.trans = {}
        self.queue = {}
        self.trans_dict = read_trans_dict(dataset)
        self.cluster_dict = get_cluster_dict(dataset)
        print type(self.sitemap)


    # @ input: page object and self.sitemap
    # @ return: cluster id
    def classify(self,file_path):
        #self.sitemap
        page = Page(file_path)
        x = []
        for feat in self.sitemap.features:
            if feat in page.xpaths:
                # remember log !
                x.append(math.log(float(page.xpaths[feat]+1),2) * pages.idf[feat])
            else:
                x.append(0.0)
        pred_y = sitemap.nbrs.predict(x)[0]
        return pred_y

    # @ input: sitemap and (page,xpath)
    # @ (cluster,xpath) - > which cluster?
    def get_xpath_transition(self):
        trans_dict = read_trans_dict(self.dataset)  # [page][xpath] = [url list] ->[cluster][xpath] = {probability list}



'''
    def crawling(self,link):
        parse the file and extract link
        get destination url
        classify page
        use self.trans to maintain self.queue





    # @ input transition matrix file
    # @ output two diction auth and hub: key: cluster_id , value: score
    def calculate_hits(self):


    def sort_queue(self):

    '''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets",help="The dataset to crawl")
    parser.add_argument('entry', help='The entry page')
    parser.add_argument('prefix', help='For urls only have partial path')
    args = parser.parse_args()
    c = crawler(args.datasets,args.entry,args.prefix)
    pages = c.sitemap.UP_pages
    c.sitemap.DBSCAN()
    sitemap = c.sitemap

    for index, page in enumerate(pages.pages):
        print page.path, pages.category[index]


    folder_path = "../../Crawler/Mar15_samples/{}/".format(args.datasets.replace("new_",""))
    gold_file = open("../Annotation/site.gold/{0}/{0}.gold".format(args.datasets.replace("new_","")),"r").readlines()
    gold_dict = {}
    for line in gold_file:
        [key,label] = line.strip().split("\t")
        gold_dict[key] = label

    right = 0
    total = 0

    for file in os.listdir(folder_path):
        total += 1
        label = gold_dict[file]
        file_path = folder_path + file
        print file_path
        page = Page(file_path)
        x = []
        for feat in sitemap.features:
            if feat in page.xpaths:
                # remember log !
                x.append(math.log(float(page.xpaths[feat]+1),2) * pages.idf[feat])
            else:
                x.append(0.0)
        pred_y = sitemap.nbrs.predict(x)[0]
        if pred_y == int(label):
            right += 1
            dist,index = sitemap.nbrs.kneighbors(x)
            #print index[0]
            #for id in index[0]:
            #    print pages.pages[id].path, pages.ground_truth[id], pages.category[id]
            #print pred_y,label

        else:
            dist,index = sitemap.nbrs.kneighbors(x)
            print index[0]
            for id in index[0]:
                print pages.pages[id].path, pages.ground_truth[id], pages.category[id]
            print pred_y,label
    print right, total



    #file_path = "../../Crawler/Mar15_samples/stackexchange/http:__android.stackexchange.com_users_263_steffen-opel.html"
    '''
    file_path = "../../Crawler/Mar15_samples/stackexchange/http:__android.stackexchange.com_a_28782.html"

    page = Page(file_path)
    print sitemap.nbrs
    print page.path
    x = []
    for feat in sitemap.features:
        if feat in page.xpaths:
            x.append(page.xpaths[feat] * pages.idf[feat])
        else:
            x.append(0.0)

    print sitemap.nbrs.predict(x)
    '''