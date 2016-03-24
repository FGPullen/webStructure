import re
import os

'''
Input: @pages:a list of list, where each list contains the split results of "_" for each html url
       @ prefix_id: assume we start to discriminate from prefix_id(start from 0)
       @ self_c_index
       @ max_c_index: we already have max_c_index clusters and we add clusters from max_c_index+1(start from 0)
Ouput: the clustering id for each file
method: first cluster based on length and then iterate each level to search for wildcard or more clusters
'''
def prefix_url_clustering(pages,clusters, prefix_id, self_c_index, max_c_index):
    prefix_dict = {}
    for index, page in enumerate(pages):
        if clusters[index] == self_c_index:
            try:
                prefix = page[prefix_id]
            except:
                print page
            if "?" in prefix and "=" in prefix:
                q_index = prefix.index("?")
                prefix = prefix[:q_index+1]
            if prefix in prefix_dict:
                clusters[index] = prefix_dict[prefix]
            else:
                if len(prefix_dict.keys())==0: ## if it is the first prefix format, stay the original cluster id
                    prefix_dict[prefix] = self_c_index
                else:
                    max_c_index += 1
                    prefix_dict[prefix] = max_c_index

    output(pages,clusters)
    print prefix_dict


def output(pages,clusters):
    for index in xrange(len(pages)):
        print "/".join(pages[index]), str(clusters[index])


def get_ground_truth(url_list, rules):
    class_list = []
    for url in url_list:
        flag = 0
        for index,rule in enumerate(rules):
            #print url,rule
            if match(url,rule):
                class_list.append(index)
                flag = 1
                break
        if flag == 0:
            class_list.append(-1)
    assert len(class_list) == len(url_list)
    return  class_list

def match(url, rule):
    strip_url = url.strip()
    temp, terms = strip_url.split("_"), []
    for term in temp:
        if term != "":
            terms.append(term)
    match_id = 0
    for index,term in enumerate(terms):
        if rule[match_id][0]=="^" and rule[match_id][-1] == "$":
            try:
                if re.match(rule[match_id],term):
                    match_id += 1
            except:
                print rule[match_id]
        else:
            if term == rule[match_id]:
                match_id += 1
        if match_id >= len(rule):
            break

    if match_id >= len(rule):
        return True
    else:
        return False

stackexchange_rules = [["a","^[0-9]+.html$"],["feeds"],["help","badges"],["help","priviledges"],["posts","^[0-9]+$", "edit.html"] , ["posts","^[0-9]+$","revisions.html"],\
["q","^[0-9]+.html$"],["questions","^[0-9]+$"],["questions","tagged"], ["revisions","view-source.html"], ["^search?(.*)$"], ["tags"],["users","^[0-9]+(.*)$"],\
["users","^signup?(.*)$"]]
rotten_rules = [["browse"],["celebrity","pictures"],["celebrity"],["critic"],["critics"],["guides"],["m"+"^[0-9]+$"+"pictures"],["m","trailers"],["m","reviews"],\
["m"],["tv"+"^[0-9]+$"+"pictures"],["tv","trailers"],["tv","reviews"],\
["tv"],["showtimes"],["^source-[0-9]+$"],["top"],["user","^[0-9]+$"]]

asp_rules = [["^[0-9]+.aspx$"],["f","rss"],["f","topanswerers"],["f"],["login","^RedirectToLogin?(.*)$"],["members"],["p","^[0-9]+$"]\
             ,["post","^[0-9]+.aspx.html$"],["private-message"],["^search?(.*)$"],["t","^[0-9]+.aspx(.*)$"],["t","next","^[0-9]+.html$"],["t","prev","^[0-9]+.html$"]\
             ,["t","rss","^[0-9]+.html$"]]

if __name__ == "__main__":
    data_folder = "./site.sample/"
    write_folder = "./site.gold/"
    datasets = ["asp"]
    read_suffix = "_urls.txt.sample.sample"
    write_suffix = ".txt.clusters"

    for dataset in datasets:
        pages = []
        url_lines = open(data_folder + dataset + read_suffix, "r").readlines()
        num_cluster = 0
        '''
        for url in file_lines:
            line = line.strip().replace(".html", "")
            temp, terms = line.split("_"), []
            for term in temp:
                if term != "":
                    terms.append(term)
            pages.append(terms)
        results = [0 for i in xrange(len(pages))]
        prefix_url_clustering(pages,results,2,0,num_cluster)
        print num_cluster
        break
        '''

        class_list = get_ground_truth(url_lines,asp_rules)


        folder = write_folder+"/"+dataset
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path= folder+"/"+dataset + ".class"
        write_file = open(file_path,"w")
        for index in xrange(len(class_list)):
            write_file.write(url_lines[index].strip() + " " + str(class_list[index]) + "\n")
            if class_list[index]==-1:
                print url_lines[index].strip() + " " + str(class_list[index])

        break