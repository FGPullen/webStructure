from pages import allPages
from collections import defaultdict

class baseline:
    def __init__(self, pages):
        self.pages = pages
        self.dt = 0.20


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

    def run(self):
        self.pattern = defaultdict(list)  # key is the id of schema and value is the list of index of pages
        self.diction = defaultdict(int) # string -> int
        self.mapping = defaultdict() # int -> set
        num = 0
        for index,page in enumerate(self.pages.pages):
            schema = str(page.anchor_xpath_set)
            if schema not in self.diction:
                self.diction[schema] = num
                self.mapping[num] = page.anchor_xpath_set
                id = num
                num += 1
            else:
                id = self.diction[schema]
            self.pattern[id].append(index)
        cardinality = {}
        for key in self.pattern:
            cardinality[key] = len(self.pattern[key])
        sorted_card,sorted_index = sorted(cardinality.iteritems(),key=lambda d:d[1],reverse=True),[]
        self.sorted_index = sorted_index
        for i,value in enumerate(sorted_card):
            sorted_index.append(value[0])
        print sorted_index,"sorted index"
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
                    self.mapping[id1]|= self.mapping[id2]
                    #print len(self.mapping[id1]), "collapse"
                    self.mapping[id2] = set()


        length = len(self.pages.ground_truth)
        print length, "length"
        self.pages.category = [0 for i in range(length)]


    # @input:  self.pattern[] int -> set of page
    #          self.mapping[] int -> set of xpath
    #          self.pages.anchor_xpath_dict
    # @output: self.pattern: combining small with large following MDL
    def MDL(self):
        self.Model = [] # only sav id
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
                    s = self.mapping[pattern_id]
                    length = len(s)
                    count = 0
                    for pid in self.pattern[id]:
                        page = self.pages.pages[pid]
                        for xpath in page.anchor_xpath_dict:
                            if xpath in s:
                                count += 1
                                encode_cost += (0.8+page.anchor_xpath_dict[xpath])
                            else:
                                encode_cost += (1+page.anchor_xpath_dict[xpath])
                        encode_cost += 2*(length-count)
                    if encode_cost < min_cost:
                        min_cost = encode_cost
                        cid = id


                self_model_cost = len(self.mapping[id])
                self_encode_cost = 0
                for xpath in page.anchor_xpath_dict:
                    self_encode_cost +=  (0.8 + max(1,page.anchor_xpath_dict[xpath]))
                self_cost = self_model_cost + self_encode_cost

                if min_cost < self_cost:
                    self.pattern[cid] += self.pattern[id]
                    self.pattern[id] = []
                else:
                    self.Model.append(id)



    def clustering(self):
        count = 0
        for key,value in self.pattern.iteritems():
            if len(value) != 0:
                print key,len(value),value
                for id in self.pattern[key]:
                    print id
                    print self.pages.pages[id].path,
                    self.pages.category[id] = count
                count += 1
                print "\n"
        print count, "number of Class"



if __name__ == "__main__":
    dataset = "youtube"
    data_pages = allPages(["../Crawler/July30_samples/{}/".format(dataset)],dataset,date="May1",mode="c_baseline")
    c_baseline = baseline(data_pages)
    print data_pages.ground_truth
    c_baseline.run()
    print " === MDL ++++"
    #c_baseline.MDL()
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