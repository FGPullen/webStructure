import os

meta_sample_folder = "./site.sample/"

for meta_file in os.listdir(meta_sample_folder):
    if meta_file == ".DS_Store":
        continue
    dataset = meta_file.split("_")[0]

    full_dataset = "../Mar15_data/" + str(dataset)
    sample_dataset = "../Mar15_samples/" + str(dataset)
    if not os.path.exists(sample_dataset):
        os.makedirs(sample_dataset)
    else:
        print sample_dataset
        print "!!!"

    meta_file = meta_sample_folder + "/" + meta_file
    for line in open(meta_file,"r").readlines():
        line = full_dataset + "/" + line