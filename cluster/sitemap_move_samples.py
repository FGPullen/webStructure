import os,sys
from shutil import copyfile
import traceback

meta_sample_folder = "../Crawler/July30_samples/"
sitemap_size = int(sys.argv[1])
dataset_list = ["asp","stackexchange","douban","youtube","douban","hupu","rottentomatoes"]


for dataset in dataset_list:

    source_path = meta_sample_folder + dataset
    target_path ="../Crawler/July30_samples/{0}/{1}".format(sitemap_size,dataset)
    if not os.path.exists("../Crawler/July30_samples/{0}".format(sitemap_size)):
        os.mkdir("../Crawler/July30_samples/{0}".format(sitemap_size))
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    file_path = "./July30/site.sample/{}.sample".format(dataset)

    lines = open(file_path,"r").readlines()
    urls = [line.strip().split()[0] for line in lines][:sitemap_size]

    for url in urls:
        file_name = url.replace("/","_") + ".html"
        source_file = source_path + "/" + file_name
        target_file = target_path + "/" + file_name
        try:
            copyfile(source_file, target_file)
        except:
            print source_file
            traceback.print_exc()
