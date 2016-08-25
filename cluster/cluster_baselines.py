from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold
from pages import allPages
import pickle,sys
import numpy as np
from pageCluster import pageCluster

class baseline:
    def __init__(self, pages, dataset):
        self.pages = pages
        self.dt = 0.20
        self.dataset = dataset
        self.date = "July30"
        self.cluster_dict = self.read_dbscan_results()
        self.length = len(self.pages.ground_truth)
        print len(self.cluster_dict)
        #raise

    def read_dbscan_results(self):
        file_path = open("./July30/site.dbscan/{}.txt".format(self.dataset),"r")
        cluster_dict = {}
        for line in file_path.readlines():
            line = line.strip()
            try:
                temp = line.split()
                gold = temp[-2]
                cluster = temp[-1]
                name = " ".join(temp[:-2]).replace(".html","")

            except:
                print line
            gold = int(gold.replace("gold:",""))
            cluster = int(cluster.replace("cluster:",""))
            cluster_dict[name] = cluster
        #for path in self.pages.path_list:
        #    print path, cluster_dict[path]
        return cluster_dict



    def jaccard(self,page1,page2):
        union = len(page1 | page2)
        inter = len(page1 & page2)
        #union = len(page1.anchor_xpath_set|page2.anchor_xpath_set)
        #inter = len(page1.anchor_xpath_set&page2.anchor_xpath_set)
        if union == 0:
            distance = 0
        else:
            distance = 1 - float(inter)/float(union)
        return distance

    def run(self, index_list=None):
        if index_list is None:
            index_list = [i for i in range(self.length)]
        self.pattern = defaultdict(list)  # key is the id of schema and value is the list of index of pages
        self.diction = defaultdict(int) # string -> int
        self.mapping = defaultdict() # int -> set
        length = len(index_list)
        print length, "length"
        self.pages.category = [-1 for i in range(self.length)]
        self.pages.dbscan = [-1 for i in range(self.length)]
        num = 0
        # id is the #id schema that appears
        for index in index_list:
            page = self.pages.pages[index]
            path = page.path
            self.pages.dbscan[index] = self.cluster_dict[path]
            schema = str(page.anchor_xpath_set)
            if schema not in self.diction:
                self.diction[schema] = num
                self.mapping[num] = page.anchor_xpath_set
                id = num
                num += 1
            else:
                id = self.diction[schema]
            self.pattern[id].append(index)
            print path, 'path in index_list'
        cardinality = {}
        for key in self.pattern:
            cardinality[key] = len(self.pattern[key])
        sorted_card,sorted_index = sorted(cardinality.iteritems(),key=lambda d:d[1],reverse=True),[]

        for i,value in enumerate(sorted_card):
            sorted_index.append(value[0])
        self.sorted_index = sorted_index

        # sort by cardinality
        for i in range(len(sorted_index)):
            for j in range(len(sorted_index)-1,i,-1):
                id1,id2 = sorted_index[i],sorted_index[j]
                if self.pattern[id1] == [] or self.pattern[id2] == []:
                    continue
                s1,s2 = self.mapping[id1],self.mapping[id2]
                if self.jaccard(s1,s2) < self.dt:
                    self.pattern[id1] += self.pattern[id2]
                    self.pattern[id2] = []
                    # collapsing small into large including schema
                    #print len(self.mapping[id1]),
                    #self.mapping[id1]|= self.mapping[id2]
                    #print len(self.mapping[id1]), "collapse"
                    #self.mapping[id2] = set()

        print self.mapping, "cluster mapping path set"



    # @input:  self.pattern[] int -> set of page
    #          self.mapping[] int -> set of xpath
    #          self.pages.anchor_xpath_dict
    # @output: self.pattern: combining small with large following MDL
    def MDL(self):
        self.Model = [] # only sav id
        cardinality = {}
        for key in self.pattern:
            cardinality[key] = len(self.pattern[key])
        sorted_card,sorted_index = sorted(cardinality.iteritems(),key=lambda d:d[1],reverse=True),[]

        for i,value in enumerate(sorted_card):
            sorted_index.append(value[0])
        self.sorted_index = sorted_index
        #print "sorted after combine"
        #for key in sorted_index:
        #    print key, cardinality[key]

        for id in self.sorted_index:
            if self.Model == []:
                self.Model = [id]
            else:
                # combine part
                min_cost,cid = 999999999,-1
                model_cost = 0
                for i in range(len(self.Model)):
                    encode_cost = 0.0
                    pattern_id = self.Model[i]
                    # s is the pattern , type is a set of string
                    s = self.mapping[pattern_id]

                    for pid in self.pattern[id]:
                        page = self.pages.pages[pid]
                        length = len(page.anchor_xpath_dict)
                        count = 0
                        for xpath in page.anchor_xpath_dict:
                            if xpath in s:
                                count += 1
                                encode_cost += (0.8+page.anchor_xpath_dict[xpath])
                            else:
                                # miss cost more
                                encode_cost += (1.0+page.anchor_xpath_dict[xpath])
                        #print length,count, "total xpath and existed xpath"
                        encode_cost += 2*(length-count)
                    if encode_cost < min_cost:
                        min_cost = encode_cost
                        cid = pattern_id

                self_model_cost = len(self.mapping[id])
                self_encode_cost = 0
                for pid in self.pattern[id]:
                    page = self.pages.pages[pid]
                    for xpath in page.anchor_xpath_dict:
                        self_encode_cost +=  (0.8 + max(1,page.anchor_xpath_dict[xpath]))
                self_cost = self_model_cost + self_encode_cost

                print self.Model, "current model list"
                print min_cost, self_cost, "min combing cost and self cost"

                if min_cost < self_cost:
                    self.pattern[cid] += self.pattern[id]
                    self.pattern[id] = []
                    print self.mapping[cid],"large set"
                    print self.mapping[id], "small one"
                else:
                    self.Model.append(id)



    def clustering(self):
        count = 0
        for key,value in self.pattern.iteritems():
            if len(value) != 0:
                #print key,len(value),value
                for id in self.pattern[key]:
                    #print self.pages.pages[id].path, count, self.pages.dbscan[id], " results "
                    if self.pages.dbscan[id] == -1:
                        self.pages.category[id] = -1
                    else:
                        self.pages.category[id] = count
                count += 1
                #print "\n"
        print count, "number of Class"


    def cv(self):
        labels_true = np.array(self.pages.ground_truth)
        skf = StratifiedKFold(labels_true, n_folds=4)
        results = []
        count = 0
        p = pageCluster(self.dataset,self.date)
        for train, test in skf:
            #print train, test
            count += 1
            print "this is the {} times for CV".format(count)
            train_gold, test_gold = labels_true[train], labels_true[test]
            self.run(train)
            self.MDL()
            path_list = self.pages.path_list

            self.classify(test)
            self.clustering()

            print train, "train index list", type(train), len(train)
            train_y = np.array(self.pages.category)[train]
            test_y = np.array(self.pages.category)[test]

            results.append(p.Evaluation_CV(test_gold,test_y, train_gold, train_y, path_list=path_list))

            '''
            t = KMeans()
            train_y, final_centroids, final_ite, final_dist = t.k_means(km_train_x, num_clusters, replicates=20)
            test_y = t.k_means_classify(test_x)
            path_list = [self.UP_pages.path_list[idx] for idx in test]
            results.append(self.Evaluation_CV(test_gold,test_y,km_train_gold,train_y, path_list=path_list))
            '''
        result = np.mean(results,axis=0)
        cv_batch_file = open("./results/c_cv_baseline.results","a")
        algo = "dbscan"
        dataset = self.dataset
        prefix =  str(dataset)  +  " classifying \t"
        metrics = ['cv_micro_precision','cv_macro_precision',"non outlier ratio"]
        for index,metric in enumerate(metrics):
            line =  prefix + "\t" + metric + "\t" + str(result[index])
            print line
            cv_batch_file.write(line + "\n" )

    # using self.mapping to classify into id and add to self.pattern[id]
    def classify(self,test_list):
        for index in test_list:
            page = self.pages.pages[index]
            path = page.path
            self.pages.dbscan[index] = self.cluster_dict[path]
            schema = page.anchor_xpath_set
            min_jaccard, cid  = 1000, -1
            for id in self.mapping:
                if len(self.pattern[id]) != 0:
                    s = self.mapping[id]
                    print type(schema), " type of schema"
                    print type(s), " type of s"
                    tmp = self.jaccard(schema,s)
                    if  tmp < min_jaccard:
                        min_jaccard = tmp
                        cid = id
            self.pattern[cid].append(index)





if __name__ == "__main__":
    dataset = sys.argv[1]
    mode = sys.argv[2]
    #dataset = "rottentomatoes"
    data_pages = allPages(["../Crawler/July30_samples/{}/".format(dataset)],dataset,date="July30",mode="c_baseline")
    #with open("cluster_temp","wb") as outfile:
    #    pickle.dump(outfile,data_pages)
    c_baseline = baseline(data_pages,dataset)
    print data_pages.ground_truth
    if mode == "cv":
        c_baseline.cv()
    elif mode == "train":
        c_baseline.run()
        c_baseline.MDL()
        c_baseline.clustering()
        c_baseline.pages.Evaluation()

    '''
    for page in c_baseline.pages.pages:
        print page.anchor_xpath_set

    pages = c_baseline.pages.pages
    for i in range(len(pages)):
        page = pages[i]
        index = i
        min_d,distance = 1.0,0.0
        for j in range(len(pages)):
            if i == j:
                continue
            page2 = pages[j]
            distance = c_baseline.jaccard(page,page2)
            if distance < min_d:
                index = j
                min_d = distance
        print i,index,min_d
        print pages[i].path, pages[i].anchor_xpath_set
        print pages[index].path, pages[index].anchor_xpath_set
        print len(pages[i].anchor_xpath_set|pages[index].anchor_xpath_set),len(pages[i].anchor_xpath_set&pages[index].anchor_xpath_set)
        '''