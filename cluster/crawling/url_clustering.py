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

douban_rules = [["awards"],["celebrity","^[0-9]+$"],["feed","subject","^[0-9]+$"],["photos","photo"],["review","^[0-9]+$"],\
                ["subject","^[0-9]+$"],["ticket","^[0-9]+$"],["trailer","^[0-9]+$"]]

youtube_rules = [["channel","^(.*).html$"],["^playlist\?list=(.*)$"],["user","^playlists(.*).html$"],["user","^videos(.*).html$"],["user","^discussion(.*).html$"],["user","^(.*).html$"],["^watch\?v=(.*)$"],["channels","^(.*).html$"]]

tripadvisor_rules = [["^AllLocations(.*)$"],["^Attractions(.*)$"],["^Flights(.*)$"],["Hotel","^Review-(.*)$"],["^Hotels-(.*)$"],["^HotelsList-(.*)$"],["^HotelsNear-(.*)$"]\
                     ,["^LastMinute(.*)$"],["^LocalMaps(.*)$"],["^Offers(.*)$"],["^Restaurants(.*)$"],["^ShowForum(.*)$"],\
                     ["^ShowUserReviews(.*)$"],["^Tourism(.*)$"],["Travel","^Guide(.*)$"],["^TravelersChoice(.*)$"],["^VacationRentals(.*)$"],["UserReview-e"]]

biketo_rules = [["^forum-[0-9]+-[0-9]+(.*)$"],["^space-uid-[0-9]+(.*)$"],["^thread-[0-9]+-(.*)$"],["columns","^[0-9]+$"],["daily","^[0-9]+(.*)$"],\
               ["c?aid=[0-9]+(.*)"],["edge","beginner","index"],["edge","health","^[0-9]+$(.*)"],["edge","knowledge","^[0-9]+$(.*)"],\
                ["edge","Photographic","^[0-9]+(.*)$"],["edge","repair","^[0-9]+(.*)$"],["edge","safe","^[0-9]+(.*)$"],["Gallery","photograph","^[0-9]+(.*)$"],\
                ["industry","business","^[0-9]+(.*)$"],["industry","cover","^[0-9]+(.*)$"],["industry","exhibition","^[0-9]+(.*)$"],["info"],["news","activity","^[0-9]+(.*)$"],["news","bikenews","^[0-9]+(.*)$"],\
                ["news","girl","^[0-9]+(.*)$"],["international","^[0-9]+(.*)$"],["news","picture","^[0-9]+(.*)$"],["product","bikes","index"],["product","bikes","^[0-9]+(.*)$"],["product","equipment","^[0-9]+(.*)$"],\
                ["gearest","^[0-9]+(.*)$"],["racing","column","^[0-9]+(.*)$"],["racing","cover"],["racing","Events","^[0-9]+(.*)$"],["racing","herald","^[0-9]+(.*)$"],\
                ["racing","index"],["racing","internal","^[0-9]+(.*)$"],["racing","news","^[0-9]+(.*)$"],["s"],["z","all","^[0-9]+(.*)$"],["z","^[0-9]+(.*)$"]]


hupu_rules = [["cba","^[0-9]+(.*)$"],["china","^[0-9]+(.*)$"],["f1","^[0-9]+(.*)$"],["other","^[0-9]+(.*)$"],["soccer","^[0-9]+(.*)$"],["sports","^[0-9]+(.*)$"],\
              ["tennis","^[0-9]+(.*)$"],["zb","^[0-9]+(.*)$"],["nba","^[0-9]+(.*)$"],["o"],["people","^[0-9]+(.*)$"]]


baidu_rules = [["bawu2","platform","^detailsInfo(.*)$"],["bawu2","platform","^listMemberInfo(.*)$"],["^f\?ie=utf-8(.*)$"],["f","^good\?kw(.*)$"],["f","index"],["^f\?kw(.*)$"],["f","like"],["game","^index?(.*)$"],["home","^main(.*)$"],["p","^[0-9]+(.*)$"],["shipin","bw"],\
               ["sign","^index(.*)$"],["tousu","new","^add(.*)$"]]


if __name__ == "__main__":
    data_folder = "./May1/site.sample/"
    write_folder = "./May1/site.gold/"
    datasets = ["stackexchange","youtube","asp","tripadvisor","douban"]
    rules = [stackexchange_rules,youtube_rules,asp_rules,tripadvisor_rules,douban_rules]
    read_suffix = ".sample"
    write_suffix = ".txt.clusters"

    for index, dataset in enumerate(datasets):
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

        class_list = get_ground_truth(url_lines,rules[index])

        folder = write_folder+"/"+dataset
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path= folder+"/"+dataset + ".class"
        write_file = open(file_path,"w")
        for index in xrange(len(class_list)):
            write_file.write(url_lines[index].strip() + " " + str(class_list[index]) + "\n")
            if class_list[index] ==-1:
                print url_lines[index].strip() + " " + str(class_list[index])

        #break

    #os.system("python combine_clustering.py")