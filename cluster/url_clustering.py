import os




if __name__ == "__main__":
    data_path = "./Data/site.sample"

    for file in os.listdir(data_path):
        if file == ".DS_store":
            continue
        write_path = "./Data/ground_truth/" + file.replace("_urls.txt.sample.sample",".url.gold")
        write_file = open(write_path,"w")
        url_lines = open(file,"r").readlines()
        for line in url_lines:
            print line

