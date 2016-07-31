class cluster:
    def __init__(self, pages):
       self.schema = set()


    #def init_cluster_from_schema(self,pages):


    '''
    def addPage(self,page_object):
        self.pages.append(page_object)

    def find_local_stop_structure(self,global_nidf,global_threshold):
        self.local_nidf = {}
        for xpath in global_nidf:
            if global_nidf[xpath] < global_threshold:
                self.local_nidf[xpath] = 0
                for page in self.pages:
                    if page.xpaths[xpath] > 0 :
                        self.local_nidf[xpath] += 1
        print len(self.local_nidf)
        local_sorted_list = sorted(self.local_nidf.iteritems(), key=lambda d:d[1], reverse =True)
        for i in range(20):
            print str(local_sorted_list[i][1]) + "\t" + local_sorted_list[i][0] + "\t" + str(global_nidf[local_sorted_list[i][0]])

    '''
