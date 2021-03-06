import os
from sets import Set
if __name__ == "__main__":
    write_folder = "./May1/site.gold/"
    datasets = ["stackexchange","asp","douban","youtube","tripadvisor","hupu","baidu","biketo"]
    for dataset in datasets:
        gold_class_list = []
        class_file = write_folder + dataset + "/" + dataset + ".class"
        combine_file = write_folder + dataset + "/" + dataset + ".combine"
        final_file = write_folder + dataset + "/" + dataset + ".gold"
        write_file = open(final_file,"w")
        map_dict = {}
        for line in open(combine_file,"r").readlines():
            line = line.strip()
            print line
            if line!="":
                [id1,id2,comment] = line.split()
                map_dict[int(id2)] = int(id1)

        for line in open(class_file,"r").readlines():
            line = line.strip()
            if line!="":
                print line
                tmp = line.split()
                class_id = tmp[-1]
                path = "".join(tmp[:-1])
                class_id = int(class_id)
                if class_id in map_dict:
                    class_id = map_dict[class_id]
                gold_class_list.append(class_id)
                path = path.replace("_","/")
                write_file.write(path+"\t"+str(class_id)+"\n")

        print len(Set(gold_class_list))